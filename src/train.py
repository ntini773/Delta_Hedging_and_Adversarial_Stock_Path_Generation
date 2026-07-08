from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch

from src.data import FEATURE_BUILDERS, prepare_dataset_bundle, to_torch
from src.loss import calculate_cvar, calculate_cvar_loss, calculate_downside_deviation, calculate_hedging_pnl, calculate_var
from src.model import MODEL_SPECS, build_model, run_deep_hedger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a minimal PyTorch deep hedger.")
    parser.add_argument("--model-version", default="v1", choices=sorted(MODEL_SPECS.keys()))
    parser.add_argument("--csv-path", default="data/20260205_option_minute_prices_expiry.csv")
    parser.add_argument("--expiry-time", default="2026-02-05 15:30:00")
    parser.add_argument("--regime", default="gbm", choices=["gbm", "jump_diffusion"])
    parser.add_argument("--num-paths", type=int, default=1000)
    parser.add_argument("--volatility", type=float, default=0.6)
    parser.add_argument("--rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="artifacts/checkpoints")
    return parser.parse_args()


def evaluate_model(
    model: torch.nn.Module,
    model_version: str,
    dataset: dict,
    strike: float,
    option_type: str,
    transaction_cost_rate: float,
    device: str,
) -> dict:
    model.eval()
    with torch.no_grad():
        paths = to_torch(dataset["paths"], device=device)
        bs_deltas = to_torch(dataset["bs_deltas"], device=device)
        time_grid = to_torch(dataset["time_grid"], device=device)
        implied_vol = torch.full_like(time_grid, fill_value=0.6)
        hedge_paths = run_deep_hedger(
            model=model,
            feature_builder=FEATURE_BUILDERS[model_version],
            price_paths=paths,
            bs_deltas=bs_deltas,
            time_to_expiry=time_grid,
            implied_volatility=implied_vol,
            strike=strike,
        )
        pnl, costs = calculate_hedging_pnl(
            price_paths=paths,
            hedge_paths=hedge_paths,
            strike=strike,
            option_type=option_type,
            transaction_cost_rate=transaction_cost_rate,
        )
    return {
        "pnl": pnl.cpu(),
        "transaction_costs": costs.cpu(),
        "metrics": {
            "mean_pnl": float(pnl.mean().item()),
            "std_pnl": float(pnl.std(unbiased=False).item()),
            "var_5": calculate_var(pnl, alpha=0.05),
            "cvar_5": calculate_cvar(pnl, alpha=0.05),
            "total_transaction_cost": float(costs.sum().item()),
            "downside_deviation": calculate_downside_deviation(pnl),
        },
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_bundle = prepare_dataset_bundle(
        csv_path=args.csv_path,
        expiry_time=args.expiry_time,
        regime=args.regime,
        num_paths=args.num_paths,
        volatility=args.volatility,
        rate=args.rate,
        seed=args.seed,
        test_ratio=args.test_ratio,
    )
    spec = MODEL_SPECS[args.model_version]
    device = args.device

    model = build_model(args.model_version).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_paths = to_torch(dataset_bundle["train"]["paths"], device=device)
    train_bs_deltas = to_torch(dataset_bundle["train"]["bs_deltas"], device=device)
    train_time_grid = to_torch(dataset_bundle["train"]["time_grid"], device=device)
    train_implied_vol = torch.full_like(train_time_grid, fill_value=args.volatility)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)

    for epoch in range(args.epochs):
        permutation = torch.randperm(train_paths.shape[0], generator=generator)
        epoch_losses = []
        for start in range(0, train_paths.shape[0], args.batch_size):
            batch_indices = permutation[start : start + args.batch_size]
            batch_paths = train_paths[batch_indices]
            batch_bs_deltas = train_bs_deltas[batch_indices]
            batch_time_grid = train_time_grid[batch_indices]
            batch_implied_vol = train_implied_vol[batch_indices]

            hedge_paths = run_deep_hedger(
                model=model,
                feature_builder=FEATURE_BUILDERS[args.model_version],
                price_paths=batch_paths,
                bs_deltas=batch_bs_deltas,
                time_to_expiry=batch_time_grid,
                implied_volatility=batch_implied_vol,
                strike=dataset_bundle["context"]["strike"],
            )
            pnl, _ = calculate_hedging_pnl(
                price_paths=batch_paths,
                hedge_paths=hedge_paths,
                strike=dataset_bundle["context"]["strike"],
                option_type=dataset_bundle["context"]["option_type"],
                transaction_cost_rate=spec.transaction_cost_rate,
            )
            loss = calculate_cvar_loss(pnl, alpha=0.05)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        print(f"Epoch {epoch + 1:03d}/{args.epochs}: CVaR loss = {sum(epoch_losses) / len(epoch_losses):.6f}")

    train_eval = evaluate_model(
        model=model,
        model_version=args.model_version,
        dataset=dataset_bundle["train"],
        strike=dataset_bundle["context"]["strike"],
        option_type=dataset_bundle["context"]["option_type"],
        transaction_cost_rate=spec.transaction_cost_rate,
        device=device,
    )
    test_eval = evaluate_model(
        model=model,
        model_version=args.model_version,
        dataset=dataset_bundle["test"],
        strike=dataset_bundle["context"]["strike"],
        option_type=dataset_bundle["context"]["option_type"],
        transaction_cost_rate=spec.transaction_cost_rate,
        device=device,
    )

    checkpoint_base = output_dir / spec.name
    serializable_context = {
        **dataset_bundle["context"],
        "snapshot_time": str(dataset_bundle["context"]["snapshot_time"]),
        "expiry_time": str(dataset_bundle["context"]["expiry_time"]),
    }

    metadata = {
        "saved_at": datetime.utcnow().isoformat() + "Z",
        "model_name": spec.name,
        "model_version": spec.version,
        "features": spec.features,
        "hidden_dims": spec.hidden_dims,
        "transaction_cost_rate": spec.transaction_cost_rate,
        "changes_from_previous": spec.changes_from_previous,
        "dataset_config": dataset_bundle["config"],
        "context": serializable_context,
        "train_metrics": train_eval["metrics"],
        "test_metrics": test_eval["metrics"],
    }

    torch.save(
        {
            "state_dict": model.state_dict(),
            "metadata": metadata,
        },
        checkpoint_base.with_suffix(".pt"),
    )
    checkpoint_base.with_suffix(".json").write_text(json.dumps(metadata, indent=2, default=str))

    print("\nSaved checkpoint:")
    print(checkpoint_base.with_suffix(".pt"))
    print("\nTest metrics:")
    for key, value in test_eval["metrics"].items():
        print(f"  {key}: {value:.6f}")


if __name__ == "__main__":
    main()
