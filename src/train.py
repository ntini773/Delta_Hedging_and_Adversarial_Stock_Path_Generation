from __future__ import annotations

import argparse
import json
import copy
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
    parser.add_argument("--regime", default=None, choices=["gbm", "jump_diffusion"])
    parser.add_argument("--num-paths", type=int, default=1000)
    parser.add_argument("--volatility", type=float, default=0.6)
    parser.add_argument("--rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="artifacts/checkpoints")
    parser.add_argument("--run-tag", default="")
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
        bs_gammas = to_torch(dataset["bs_gammas"], device=device)
        bs_thetas = to_torch(dataset["bs_thetas"], device=device)
        bs_vegas = to_torch(dataset["bs_vegas"], device=device)
        time_grid = to_torch(dataset["time_grid"], device=device)
        implied_vol = to_torch(dataset["implied_volatility"], device=device)
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
        regime=args.regime or MODEL_SPECS[args.model_version].default_regime,
        num_paths=args.num_paths,
        volatility=args.volatility,
        rate=args.rate,
        seed=args.seed,
        test_ratio=args.test_ratio,
        validation_ratio=args.validation_ratio,
    )
    spec = MODEL_SPECS[args.model_version]
    device = args.device

    model = build_model(args.model_version).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_paths = to_torch(dataset_bundle["train"]["paths"], device=device)
    train_bs_deltas = to_torch(dataset_bundle["train"]["bs_deltas"], device=device)
    train_bs_gammas = to_torch(dataset_bundle["train"]["bs_gammas"], device=device)
    train_bs_thetas = to_torch(dataset_bundle["train"]["bs_thetas"], device=device)
    train_bs_vegas = to_torch(dataset_bundle["train"]["bs_vegas"], device=device)
    train_time_grid = to_torch(dataset_bundle["train"]["time_grid"], device=device)
    train_implied_vol = to_torch(dataset_bundle["train"]["implied_volatility"], device=device)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    best_validation_cvar = float("inf")
    best_epoch = -1
    best_state_dict = None
    best_validation_metrics = None

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
                bs_gammas=train_bs_gammas[batch_indices],
                bs_thetas=train_bs_thetas[batch_indices],
                bs_vegas=train_bs_vegas[batch_indices],
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

        validation_eval = evaluate_model(
            model=model,
            model_version=args.model_version,
            dataset=dataset_bundle["validation"],
            strike=dataset_bundle["context"]["strike"],
            option_type=dataset_bundle["context"]["option_type"],
            transaction_cost_rate=spec.transaction_cost_rate,
            device=device,
        )
        validation_cvar = validation_eval["metrics"]["cvar_5"]
        if validation_cvar < best_validation_cvar:
            best_validation_cvar = validation_cvar
            best_epoch = epoch + 1
            best_state_dict = copy.deepcopy(model.state_dict())
            best_validation_metrics = validation_eval["metrics"]

        print(
            f"Epoch {epoch + 1:03d}/{args.epochs}: "
            f"train CVaR loss = {sum(epoch_losses) / len(epoch_losses):.6f}, "
            f"validation CVaR = {validation_cvar:.6f}"
        )

    if best_state_dict is None:
        raise RuntimeError("Training finished without selecting a best validation checkpoint.")

    final_state_dict = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state_dict)

    train_eval = evaluate_model(
        model=model,
        model_version=args.model_version,
        dataset=dataset_bundle["train"],
        strike=dataset_bundle["context"]["strike"],
        option_type=dataset_bundle["context"]["option_type"],
        transaction_cost_rate=spec.transaction_cost_rate,
        device=device,
    )
    validation_eval = evaluate_model(
        model=model,
        model_version=args.model_version,
        dataset=dataset_bundle["validation"],
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
        "run_tag": args.run_tag,
        "dataset_config": dataset_bundle["config"],
        "context": serializable_context,
        "best_epoch": best_epoch,
        "selection_metric": "validation_cvar_5",
        "train_metrics": train_eval["metrics"],
        "validation_metrics": validation_eval["metrics"],
        "test_metrics": test_eval["metrics"],
    }

    last_checkpoint_base = output_dir / f"{spec.name}_last"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "metadata": metadata,
        },
        checkpoint_base.with_suffix(".pt"),
    )
    torch.save(
        {
            "state_dict": final_state_dict,
            "metadata": metadata,
        },
        last_checkpoint_base.with_suffix(".pt"),
    )
    checkpoint_base.with_suffix(".json").write_text(json.dumps(metadata, indent=2, default=str))
    last_checkpoint_base.with_suffix(".json").write_text(json.dumps(metadata, indent=2, default=str))

    print("\nSaved checkpoint:")
    print(checkpoint_base.with_suffix(".pt"))
    print(f"Best validation epoch: {best_epoch}")
    print("\nValidation metrics:")
    for key, value in validation_eval["metrics"].items():
        print(f"  {key}: {value:.6f}")
    print("\nTest metrics:")
    for key, value in test_eval["metrics"].items():
        print(f"  {key}: {value:.6f}")


if __name__ == "__main__":
    main()
