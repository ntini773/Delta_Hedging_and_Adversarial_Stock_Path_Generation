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
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--run-tag", default="")
    return parser.parse_args()


def load_checkpoint_metadata(checkpoint_path: Path) -> dict:
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)["metadata"]


def is_checkpoint_compatible(metadata: dict, args: argparse.Namespace) -> tuple[bool, str]:
    dataset_config = metadata.get("dataset_config", {})
    required_pairs = {
        "csv_path": args.csv_path,
        "expiry_time": args.expiry_time,
        "num_paths": args.num_paths,
        "volatility": args.volatility,
        "rate": args.rate,
        "seed": args.seed,
        "test_ratio": args.test_ratio,
        "validation_ratio": args.validation_ratio,
    }
    for key, expected_value in required_pairs.items():
        if dataset_config.get(key) != expected_value:
            return False, f"{key} mismatch (checkpoint={dataset_config.get(key)!r}, expected={expected_value!r})"

    if args.run_tag and metadata.get("run_tag", "") != args.run_tag:
        return False, f"run_tag mismatch (checkpoint={metadata.get('run_tag', '')!r}, expected={args.run_tag!r})"

    if not metadata.get("selection_metric"):
        return False, "missing selection_metric metadata"

    return True, ""


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
    bs_gammas = to_torch(dataset["bs_gammas"], device=device)
    bs_thetas = to_torch(dataset["bs_thetas"], device=device)
    bs_vegas = to_torch(dataset["bs_vegas"], device=device)
    time_grid = to_torch(dataset["time_grid"], device=device)
    implied_vol = to_torch(dataset["implied_volatility"], device=device)

    with torch.no_grad():
        hedge_paths = run_deep_hedger(
            model=model,
            feature_builder=FEATURE_BUILDERS[model_version],
            price_paths=paths,
            bs_deltas=bs_deltas,
            bs_gammas=bs_gammas,
            bs_thetas=bs_thetas,
            bs_vegas=bs_vegas,
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
        "Tail Loss 5%",
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


def build_improvement_table(
    rows: list[tuple[str, dict[str, float], str]],
    baselines: dict[str, dict[str, float]],
) -> str:
    direction = {
        "mean_pnl": "higher better",
        "std_pnl": "lower better",
        "var_5": "higher better",
        "cvar_5": "lower better",
        "total_transaction_cost": "lower better",
        "downside_deviation": "lower better",
    }
    headers = ["Model", "Baseline", "Mean P&L", "Std Dev", "VaR 5%", "Tail Loss 5%", "Total Tx Cost", "Downside Dev"]
    table = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        "| Metric Direction | baseline-matched | "
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
    for model_name, metrics, baseline_key in rows:
        if model_name.startswith("Black-Scholes"):
            continue
        baseline = baselines[baseline_key]
        values = []
        for metric in ["mean_pnl", "std_pnl", "var_5", "cvar_5", "total_transaction_cost", "downside_deviation"]:
            baseline_value = baseline[metric]
            if baseline_value == 0:
                values.append("n/a")
            else:
                if metric in {"std_pnl", "cvar_5", "total_transaction_cost", "downside_deviation"}:
                    change = 100.0 * (baseline_value - metrics[metric]) / abs(baseline_value)
                else:
                    change = 100.0 * (metrics[metric] - baseline_value) / abs(baseline_value)
                values.append(f"{change:.2f}%")
        table.append("| " + " | ".join([model_name, baseline_key] + values) + " |")
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
        f"- Run tag filter: `{args.run_tag or 'none'}`",
        f"- Dataset split: `train {int((1.0 - args.test_ratio - args.validation_ratio) * 100)}% / validation {int(args.validation_ratio * 100)}% / test {int(args.test_ratio * 100)}%`",
        f"- Test set size: `{int(args.num_paths * args.test_ratio)}` paths",
        f"- Random seed: `{args.seed}`",
        f"- Jump-diffusion intensity: `{25.0}` jumps/year when regime=`jump_diffusion`",
        f"- Jump mean / std in log space: `-0.02 / 0.08`",
        "",
    ]

    raw_pnl_outputs: dict[str, np.ndarray] = {}
    discovered_models: list[dict] = []

    checkpoint_paths = sorted(path for path in checkpoints_dir.glob("deep_hedger_v*.pt") if not path.name.endswith("_last.pt"))
    for checkpoint_path in checkpoint_paths:
        metadata = load_checkpoint_metadata(checkpoint_path)
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
            validation_ratio=args.validation_ratio,
        )

        summary_rows: list[tuple[str, dict[str, float]]] = []
        improvement_rows: list[tuple[str, dict[str, float], str]] = []
        bs_no_cost_metrics, baseline_pnl = evaluate_bs_baseline(
            dataset=dataset_bundle["test"],
            context=dataset_bundle["context"],
            cost_rate=0.0,
            device=args.device,
        )
        summary_rows.append(("Black-Scholes delta (no tx cost)", bs_no_cost_metrics))
        raw_pnl_outputs[f"{regime}__bs_no_tx_cost"] = baseline_pnl

        bs_cost_metrics, bs_cost_pnl = evaluate_bs_baseline(
            dataset=dataset_bundle["test"],
            context=dataset_bundle["context"],
            cost_rate=0.001,
            device=args.device,
        )
        summary_rows.append(("Black-Scholes delta (with tx cost)", bs_cost_metrics))
        raw_pnl_outputs[f"{regime}__bs_with_tx_cost"] = bs_cost_pnl
        baselines = {
            "Black-Scholes delta (no tx cost)": bs_no_cost_metrics,
            "Black-Scholes delta (with tx cost)": bs_cost_metrics,
        }

        missing_models = []
        skipped_models: list[str] = []
        for version, spec in MODEL_SPECS.items():
            checkpoint_path = checkpoints_dir / f"{spec.name}.pt"
            if not checkpoint_path.exists():
                missing_models.append(spec.name)
                continue
            metadata = load_checkpoint_metadata(checkpoint_path)
            compatible, reason = is_checkpoint_compatible(metadata, args)
            if not compatible:
                skipped_models.append(f"{spec.name}: {reason}")
                continue
            metrics, pnl, metadata = evaluate_checkpoint(
                checkpoint_path=checkpoint_path,
                dataset=dataset_bundle["test"],
                context=dataset_bundle["context"],
                device=args.device,
            )
            summary_rows.append((metadata["model_name"], metrics))
            baseline_key = (
                "Black-Scholes delta (with tx cost)"
                if metadata["transaction_cost_rate"] > 0.0
                else "Black-Scholes delta (no tx cost)"
            )
            improvement_rows.append((metadata["model_name"], metrics, baseline_key))
            raw_pnl_outputs[f"{regime}__{metadata['model_name']}"] = pnl

        report_sections.extend(
            [
                f"## Regime: {regime}",
                "",
                build_summary_table(summary_rows),
                "",
                "### Improvement vs Matched Black-Scholes baseline",
                "",
                build_improvement_table(improvement_rows, baselines),
                "",
            ]
        )
        if missing_models:
            report_sections.append(f"Missing checkpoints for this run: `{', '.join(missing_models)}`")
            report_sections.append("")
        if skipped_models:
            report_sections.append("Skipped checkpoints due to incompatible training config:")
            for line in skipped_models:
                report_sections.append(f"- {line}")
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
            metadata = load_checkpoint_metadata(checkpoint_path)
            compatible, reason = is_checkpoint_compatible(metadata, args)
            if compatible:
                report_sections.append(f"- Checkpoint status: available")
            else:
                report_sections.append(f"- Checkpoint status: skipped")
                report_sections.append(f"- Skip reason: {reason}")
            report_sections.append("- Checkpoint type: best validation CVaR checkpoint")
            report_sections.append(f"- Features: `{', '.join(metadata['features'])}`")
            report_sections.append(f"- Hidden dims: `{metadata['hidden_dims']}`")
            report_sections.append(f"- Transaction cost rate: `{metadata['transaction_cost_rate']}`")
            report_sections.append(f"- Default regime: `{metadata['dataset_config']['regime']}`")
            if metadata["dataset_config"].get("regime") == "jump_diffusion":
                report_sections.append(
                    f"- Jump params: intensity={metadata['dataset_config'].get('jump_intensity')}, "
                    f"mean={metadata['dataset_config'].get('jump_mean')}, std={metadata['dataset_config'].get('jump_std')}"
                )
            report_sections.append(f"- Run tag: `{metadata.get('run_tag', '')}`")
        report_sections.append("")

    (output_dir / "BENCHMARK.md").write_text("\n".join(report_sections))
    np.savez(output_dir / "benchmark_pnl_arrays.npz", **raw_pnl_outputs)
    (output_dir / "benchmark_manifest.json").write_text(
        json.dumps({"generated_at": datetime.utcnow().isoformat() + "Z", "arrays": sorted(raw_pnl_outputs.keys())}, indent=2)
    )


if __name__ == "__main__":
    main()
