from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from src.data import FEATURE_BUILDERS, prepare_dataset_bundle, to_torch
from src.loss import calculate_cvar, calculate_downside_deviation, calculate_hedging_pnl, calculate_var
from src.model import MODEL_SPECS, build_model, run_deep_hedger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark saved deep hedging checkpoints.")
    parser.add_argument("--checkpoints-dir", default="artifacts/checkpoints")
    parser.add_argument("--output-dir", default="artifacts/benchmark")
    parser.add_argument("--csv-path", default="data/20260205_option_minute_prices_expiry.csv")
    parser.add_argument("--expiry-time", default="2026-02-05 15:30:00")
    parser.add_argument("--num-paths", type=int, default=1000)
    parser.add_argument("--volatility", type=float, default=0.6)
    parser.add_argument("--rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def compute_metrics(pnl: torch.Tensor, costs: torch.Tensor) -> dict[str, float]:
    return {
        "mean_pnl": float(pnl.mean().item()),
        "std_pnl": float(pnl.std(unbiased=False).item()),
        "var_5": calculate_var(pnl, alpha=0.05),
        "cvar_5": calculate_cvar(pnl, alpha=0.05),
        "total_transaction_cost": float(costs.sum().item()),
        "downside_deviation": calculate_downside_deviation(pnl),
    }


def evaluate_bs_baseline(dataset: dict, context: dict, cost_rate: float, device: str) -> tuple[dict[str, float], np.ndarray]:
    paths = to_torch(dataset["paths"], device=device)
    bs_deltas = to_torch(dataset["bs_deltas"], device=device)
    pnl, costs = calculate_hedging_pnl(
        price_paths=paths,
        hedge_paths=bs_deltas,
        strike=context["strike"],
        option_type=context["option_type"],
        transaction_cost_rate=cost_rate,
    )
    return compute_metrics(pnl, costs), pnl.cpu().numpy()


def evaluate_checkpoint(checkpoint_path: Path, dataset: dict, context: dict, device: str) -> tuple[dict[str, float], np.ndarray, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    metadata = checkpoint["metadata"]
    model_version = metadata["model_version"]
    model = build_model(model_version).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    paths = to_torch(dataset["paths"], device=device)
    bs_deltas = to_torch(dataset["bs_deltas"], device=device)
    time_grid = to_torch(dataset["time_grid"], device=device)
    implied_vol = torch.full_like(time_grid, fill_value=metadata["dataset_config"]["volatility"])

    with torch.no_grad():
        hedge_paths = run_deep_hedger(
            model=model,
            feature_builder=FEATURE_BUILDERS[model_version],
            price_paths=paths,
            bs_deltas=bs_deltas,
            time_to_expiry=time_grid,
            implied_volatility=implied_vol,
            strike=context["strike"],
        )
        pnl, costs = calculate_hedging_pnl(
            price_paths=paths,
            hedge_paths=hedge_paths,
            strike=context["strike"],
            option_type=context["option_type"],
            transaction_cost_rate=metadata["transaction_cost_rate"],
        )
    return compute_metrics(pnl, costs), pnl.cpu().numpy(), metadata


def format_metric(value: float) -> str:
    return f"{value:.6f}"


def build_summary_table(rows: list[tuple[str, dict[str, float]]]) -> str:
    headers = [
        "Model",
        "Mean P&L",
        "Std Dev",
        "VaR 5%",
        "CVaR 5%",
        "Total Tx Cost",
        "Downside Dev",
    ]
    table = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for model_name, metrics in rows:
        table.append(
            "| "
            + " | ".join(
                [
                    model_name,
                    format_metric(metrics["mean_pnl"]),
                    format_metric(metrics["std_pnl"]),
                    format_metric(metrics["var_5"]),
                    format_metric(metrics["cvar_5"]),
                    format_metric(metrics["total_transaction_cost"]),
                    format_metric(metrics["downside_deviation"]),
                ]
            )
            + " |"
        )
    return "\n".join(table)


def build_improvement_table(rows: list[tuple[str, dict[str, float]]], baseline: dict[str, float]) -> str:
    direction = {
        "mean_pnl": "higher better",
        "std_pnl": "lower better",
        "var_5": "higher better",
        "cvar_5": "lower better",
        "total_transaction_cost": "lower better",
        "downside_deviation": "lower better",
    }
    headers = ["Model", "Mean P&L", "Std Dev", "VaR 5%", "CVaR 5%", "Total Tx Cost", "Downside Dev"]
    table = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        "| Metric Direction | "
        + " | ".join(
            [
                direction["mean_pnl"],
                direction["std_pnl"],
                direction["var_5"],
                direction["cvar_5"],
                direction["total_transaction_cost"],
                direction["downside_deviation"],
            ]
        )
        + " |",
    ]
    for model_name, metrics in rows:
        if model_name.startswith("Black-Scholes"):
            continue
        values = []
        for metric in ["mean_pnl", "std_pnl", "var_5", "cvar_5", "total_transaction_cost", "downside_deviation"]:
            baseline_value = baseline[metric]
            if baseline_value == 0:
                values.append("n/a")
            else:
                change = 100.0 * (metrics[metric] - baseline_value) / abs(baseline_value)
                values.append(f"{change:.2f}%")
        table.append("| " + " | ".join([model_name] + values) + " |")
    return "\n".join(table)


def main() -> None:
    args = parse_args()
    checkpoints_dir = Path(args.checkpoints_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    regimes = ["gbm", "jump_diffusion"]
    report_sections = [
        "# Benchmark Report",
        "",
        f"- Benchmark run date: `{datetime.utcnow().isoformat()}Z`",
        f"- Test set size: `{int(args.num_paths * args.test_ratio)}` paths",
        f"- Random seed: `{args.seed}`",
        "",
    ]

    raw_pnl_outputs: dict[str, np.ndarray] = {}
    discovered_models: list[dict] = []

    checkpoint_paths = sorted(checkpoints_dir.glob("*.pt"))
    for checkpoint_path in checkpoint_paths:
        metadata = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["metadata"]
        discovered_models.append(metadata)

    for regime in regimes:
        dataset_bundle = prepare_dataset_bundle(
            csv_path=args.csv_path,
            expiry_time=args.expiry_time,
            regime=regime,
            num_paths=args.num_paths,
            volatility=args.volatility,
            rate=args.rate,
            seed=args.seed,
            test_ratio=args.test_ratio,
        )

        summary_rows: list[tuple[str, dict[str, float]]] = []
        baseline_metrics, baseline_pnl = evaluate_bs_baseline(
            dataset=dataset_bundle["test"],
            context=dataset_bundle["context"],
            cost_rate=0.0,
            device=args.device,
        )
        summary_rows.append(("Black-Scholes delta (no tx cost)", baseline_metrics))
        raw_pnl_outputs[f"{regime}__bs_no_tx_cost"] = baseline_pnl

        bs_cost_metrics, bs_cost_pnl = evaluate_bs_baseline(
            dataset=dataset_bundle["test"],
            context=dataset_bundle["context"],
            cost_rate=0.001,
            device=args.device,
        )
        summary_rows.append(("Black-Scholes delta (with tx cost)", bs_cost_metrics))
        raw_pnl_outputs[f"{regime}__bs_with_tx_cost"] = bs_cost_pnl

        missing_models = []
        for version, spec in MODEL_SPECS.items():
            checkpoint_path = checkpoints_dir / f"{spec.name}.pt"
            if not checkpoint_path.exists():
                missing_models.append(spec.name)
                continue
            metrics, pnl, metadata = evaluate_checkpoint(
                checkpoint_path=checkpoint_path,
                dataset=dataset_bundle["test"],
                context=dataset_bundle["context"],
                device=args.device,
            )
            summary_rows.append((metadata["model_name"], metrics))
            raw_pnl_outputs[f"{regime}__{metadata['model_name']}"] = pnl

        report_sections.extend(
            [
                f"## Regime: {regime}",
                "",
                build_summary_table(summary_rows),
                "",
                "### Improvement vs Black-Scholes baseline",
                "",
                build_improvement_table(summary_rows, baseline_metrics),
                "",
            ]
        )
        if missing_models:
            report_sections.append(f"Missing checkpoints for this run: `{', '.join(missing_models)}`")
            report_sections.append("")

    report_sections.extend(["## Model Changes", ""])
    for version, spec in MODEL_SPECS.items():
        report_sections.append(f"### {spec.name}")
        for line in spec.changes_from_previous:
            report_sections.append(f"- {line}")
        checkpoint_path = checkpoints_dir / f"{spec.name}.pt"
        if not checkpoint_path.exists():
            report_sections.append("- Checkpoint status: missing")
        else:
            metadata = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["metadata"]
            report_sections.append(f"- Checkpoint status: available")
            report_sections.append(f"- Features: `{', '.join(metadata['features'])}`")
            report_sections.append(f"- Hidden dims: `{metadata['hidden_dims']}`")
            report_sections.append(f"- Transaction cost rate: `{metadata['transaction_cost_rate']}`")
        report_sections.append("")

    (output_dir / "BENCHMARK.md").write_text("\n".join(report_sections))
    np.savez(output_dir / "benchmark_pnl_arrays.npz", **raw_pnl_outputs)
    (output_dir / "benchmark_manifest.json").write_text(
        json.dumps({"generated_at": datetime.utcnow().isoformat() + "Z", "arrays": sorted(raw_pnl_outputs.keys())}, indent=2)
    )


if __name__ == "__main__":
    main()
