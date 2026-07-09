from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.data import FEATURE_BUILDERS, prepare_dataset_bundle, to_torch
from src.model import MODEL_SPECS, build_model, run_deep_hedger

SPARK_CHARS = " .:-=+*#%@"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rich TUI for single-path test-set inference.")
    parser.add_argument("--checkpoints-dir", default="artifacts/checkpoints")
    parser.add_argument("--checkpoint-path", default="")
    parser.add_argument("--model-version", default="v3", choices=sorted(MODEL_SPECS.keys()))
    parser.add_argument("--csv-path", default="")
    parser.add_argument("--expiry-time", default="")
    parser.add_argument("--regime", default="", choices=["", "gbm", "jump_diffusion"])
    parser.add_argument("--num-paths", type=int, default=-1)
    parser.add_argument("--volatility", type=float, default=-1.0)
    parser.add_argument("--rate", type=float, default=-1.0)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--test-ratio", type=float, default=-1.0)
    parser.add_argument("--validation-ratio", type=float, default=-1.0)
    parser.add_argument("--path-index", type=int, default=0)
    parser.add_argument("--delay", type=float, default=0.10)
    parser.add_argument("--history-window", type=int, default=10)
    parser.add_argument("--spark-width", type=int, default=60)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def resolve_checkpoint_path(args: argparse.Namespace) -> Path:
    if args.checkpoint_path:
        checkpoint_path = Path(args.checkpoint_path)
    else:
        checkpoint_path = Path(args.checkpoints_dir) / f"{MODEL_SPECS[args.model_version].name}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(checkpoint_path: Path, device: str) -> tuple[torch.nn.Module, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    metadata = payload["metadata"]
    model = build_model(metadata["model_version"]).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, metadata


def resolve_value(override, default):
    if override in ("", -1, -1.0):
        return default
    return override


def build_dataset(args: argparse.Namespace, metadata: dict) -> tuple[dict, dict]:
    dataset_config = metadata.get("dataset_config", {})
    resolved = {
        "csv_path": resolve_value(args.csv_path, dataset_config.get("csv_path", "data/20260205_option_minute_prices_expiry.csv")),
        "expiry_time": resolve_value(args.expiry_time, dataset_config.get("expiry_time", "2026-02-05 15:30:00")),
        "regime": resolve_value(args.regime, dataset_config.get("regime", metadata["model_version"] == "v3" and "jump_diffusion" or "gbm")),
        "num_paths": int(resolve_value(args.num_paths, dataset_config.get("num_paths", 1000))),
        "volatility": float(resolve_value(args.volatility, dataset_config.get("volatility", 0.6))),
        "rate": float(resolve_value(args.rate, dataset_config.get("rate", 0.05))),
        "seed": int(resolve_value(args.seed, dataset_config.get("seed", 42))),
        "test_ratio": float(resolve_value(args.test_ratio, dataset_config.get("test_ratio", 0.2))),
        "validation_ratio": float(resolve_value(args.validation_ratio, dataset_config.get("validation_ratio", 0.1))),
        "target_option_symbol": dataset_config.get("target_option_symbol"),
    }
    bundle = prepare_dataset_bundle(
        csv_path=resolved["csv_path"],
        expiry_time=resolved["expiry_time"],
        regime=resolved["regime"],
        num_paths=resolved["num_paths"],
        volatility=resolved["volatility"],
        rate=resolved["rate"],
        seed=resolved["seed"],
        test_ratio=resolved["test_ratio"],
        validation_ratio=resolved["validation_ratio"],
        target_option_symbol=resolved["target_option_symbol"],
    )
    return bundle, resolved


def ascii_sparkline(values: list[float], width: int) -> str:
    if not values:
        return ""
    if len(values) <= width:
        sampled = values
    else:
        sampled = []
        last_index = len(values) - 1
        for i in range(width):
            idx = round(i * last_index / max(1, width - 1))
            sampled.append(values[idx])

    low = min(sampled)
    high = max(sampled)
    if abs(high - low) < 1e-12:
        return "-" * len(sampled)

    chars = []
    max_index = len(SPARK_CHARS) - 1
    for value in sampled:
        scaled = (value - low) / (high - low)
        chars.append(SPARK_CHARS[int(round(scaled * max_index))])
    return "".join(chars)


def compute_running_path_metrics(
    prices: list[float],
    hedge_path: list[float],
    strike: float,
    option_type: str,
    transaction_cost_rate: float,
) -> tuple[list[float], list[float], list[float]]:
    realized_pnl = []
    realized_costs = []
    cumulative_total = []
    trading_pnl = 0.0
    transaction_costs = 0.0

    for step, price in enumerate(prices):
        previous_delta = hedge_path[step - 1] if step > 0 else 0.0
        current_delta = hedge_path[step]
        transaction_costs += transaction_cost_rate * price * abs(current_delta - previous_delta)

        if step > 0:
            trading_pnl += hedge_path[step - 1] * (prices[step] - prices[step - 1])

        total_pnl = trading_pnl - transaction_costs
        if step == len(prices) - 1:
            payoff = max(prices[-1] - strike, 0.0) if option_type == "call" else max(strike - prices[-1], 0.0)
            transaction_costs += transaction_cost_rate * prices[-1] * abs(hedge_path[-1])
            total_pnl = trading_pnl - transaction_costs - payoff

        realized_pnl.append(trading_pnl)
        realized_costs.append(transaction_costs)
        cumulative_total.append(total_pnl)

    return realized_pnl, realized_costs, cumulative_total


def build_inference_state(
    model: torch.nn.Module,
    metadata: dict,
    bundle: dict,
    resolved_config: dict,
    path_index: int,
    device: str,
) -> dict:
    test_paths = bundle["test"]["paths"]
    if path_index < 0 or path_index >= len(test_paths):
        raise IndexError(f"path_index {path_index} is outside the test set range 0..{len(test_paths) - 1}")

    single = {
        "paths": test_paths[path_index : path_index + 1],
        "bs_deltas": bundle["test"]["bs_deltas"][path_index : path_index + 1],
        "bs_gammas": bundle["test"]["bs_gammas"][path_index : path_index + 1],
        "bs_thetas": bundle["test"]["bs_thetas"][path_index : path_index + 1],
        "bs_vegas": bundle["test"]["bs_vegas"][path_index : path_index + 1],
        "time_grid": bundle["test"]["time_grid"][path_index : path_index + 1],
        "implied_volatility": bundle["test"]["implied_volatility"][path_index : path_index + 1],
    }

    with torch.no_grad():
        deep_hedge = run_deep_hedger(
            model=model,
            feature_builder=FEATURE_BUILDERS[metadata["model_version"]],
            price_paths=to_torch(single["paths"], device=device),
            bs_deltas=to_torch(single["bs_deltas"], device=device),
            bs_gammas=to_torch(single["bs_gammas"], device=device),
            bs_thetas=to_torch(single["bs_thetas"], device=device),
            bs_vegas=to_torch(single["bs_vegas"], device=device),
            time_to_expiry=to_torch(single["time_grid"], device=device),
            implied_volatility=to_torch(single["implied_volatility"], device=device),
            strike=bundle["context"]["strike"],
        )[0].cpu().tolist()

    prices = single["paths"][0].tolist()
    bs_deltas = single["bs_deltas"][0].tolist()
    bs_trading, bs_costs, bs_total = compute_running_path_metrics(
        prices=prices,
        hedge_path=bs_deltas,
        strike=bundle["context"]["strike"],
        option_type=bundle["context"]["option_type"],
        transaction_cost_rate=metadata["transaction_cost_rate"],
    )
    deep_trading, deep_costs, deep_total = compute_running_path_metrics(
        prices=prices,
        hedge_path=deep_hedge,
        strike=bundle["context"]["strike"],
        option_type=bundle["context"]["option_type"],
        transaction_cost_rate=metadata["transaction_cost_rate"],
    )

    return {
        "metadata": metadata,
        "config": resolved_config,
        "context": bundle["context"],
        "path_index": path_index,
        "global_test_index": bundle["test_indices"][path_index],
        "test_size": len(bundle["test"]["paths"]),
        "prices": prices,
        "time_to_expiry": single["time_grid"][0].tolist(),
        "implied_volatility": single["implied_volatility"][0].tolist(),
        "bs_deltas": bs_deltas,
        "deep_deltas": deep_hedge,
        "bs_trading_pnl": bs_trading,
        "bs_costs": bs_costs,
        "bs_total_pnl": bs_total,
        "deep_trading_pnl": deep_trading,
        "deep_costs": deep_costs,
        "deep_total_pnl": deep_total,
    }


def format_float(value: float) -> str:
    return f"{value:.4f}"


def build_header(state: dict, step: int, done: bool) -> Panel:
    title = Text("Deep Hedger Inference TUI", style="bold cyan")
    subtitle = Text(
        f"{state['metadata']['model_name']} | regime={state['config']['regime']} | "
        f"test path {state['path_index']} of {state['test_size'] - 1} | "
        f"dataset index {state['global_test_index']}",
        style="white",
    )
    status = Text(
        f"step {step + 1}/{len(state['prices'])} | {'complete' if done else 'streaming'}",
        style="bold green" if done else "bold yellow",
    )
    return Panel(Group(Align.center(title), Align.center(subtitle), Align.center(status)), box=box.ROUNDED)


def build_summary_table(state: dict, step: int) -> Table:
    table = Table(box=box.SIMPLE_HEAVY)
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Spot", format_float(state["prices"][step]))
    table.add_row("BS delta", format_float(state["bs_deltas"][step]))
    table.add_row("Model delta", format_float(state["deep_deltas"][step]))
    table.add_row("Delta gap", format_float(state["deep_deltas"][step] - state["bs_deltas"][step]))
    table.add_row("Time to expiry", format_float(state["time_to_expiry"][step]))
    table.add_row("Implied vol", format_float(state["implied_volatility"][step]))
    table.add_row("BS total PnL", format_float(state["bs_total_pnl"][step]))
    table.add_row("Model total PnL", format_float(state["deep_total_pnl"][step]))
    table.add_row("Model tx cost", format_float(state["deep_costs"][step]))
    return table


def build_context_table(state: dict) -> Table:
    table = Table(box=box.SIMPLE_HEAVY)
    table.add_column("Context", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Option symbol", str(state["context"]["target_option_symbol"]))
    table.add_row("Option type", str(state["context"]["option_type"]))
    table.add_row("Strike", format_float(float(state["context"]["strike"])))
    table.add_row("Transaction cost", format_float(float(state["metadata"]["transaction_cost_rate"])))
    table.add_row("Checkpoint best epoch", str(state["metadata"].get("best_epoch", "n/a")))
    table.add_row("Selection metric", str(state["metadata"].get("selection_metric", "n/a")))
    table.add_row("Device", str(state["config"].get("device", "cpu")))
    return table


def build_trajectory_panel(state: dict, step: int, spark_width: int) -> Panel:
    price_history = state["prices"][: step + 1]
    bs_history = state["bs_deltas"][: step + 1]
    deep_history = state["deep_deltas"][: step + 1]
    lines = [
        f"Price  {ascii_sparkline(price_history, spark_width)}",
        f"BS     {ascii_sparkline(bs_history, spark_width)}",
        f"Model  {ascii_sparkline(deep_history, spark_width)}",
    ]
    return Panel("\n".join(lines), title="Trajectory", box=box.ROUNDED)


def build_recent_steps_table(state: dict, step: int, history_window: int) -> Table:
    table = Table(box=box.SIMPLE_HEAVY)
    table.add_column("Step", justify="right")
    table.add_column("Spot", justify="right")
    table.add_column("BS d", justify="right")
    table.add_column("Model d", justify="right")
    table.add_column("BS PnL", justify="right")
    table.add_column("Model PnL", justify="right")

    start = max(0, step - history_window + 1)
    for row_step in range(start, step + 1):
        table.add_row(
            str(row_step),
            format_float(state["prices"][row_step]),
            format_float(state["bs_deltas"][row_step]),
            format_float(state["deep_deltas"][row_step]),
            format_float(state["bs_total_pnl"][row_step]),
            format_float(state["deep_total_pnl"][row_step]),
        )
    return table


def build_footer_panel(state: dict, step: int) -> Panel:
    delta_gap = state["deep_deltas"][step] - state["bs_deltas"][step]
    pnl_gap = state["deep_total_pnl"][step] - state["bs_total_pnl"][step]
    text = Text()
    text.append("Model vs BS: ", style="bold")
    text.append(f"delta gap {delta_gap:.4f}", style="cyan")
    text.append(" | ")
    text.append(f"PnL gap {pnl_gap:.4f}", style="green" if pnl_gap >= 0 else "red")
    return Panel(text, box=box.ROUNDED)


def render_dashboard(state: dict, step: int, history_window: int, spark_width: int) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=5),
        Layout(name="body", ratio=1),
        Layout(name="footer", size=3),
    )
    layout["body"].split_row(Layout(name="left", ratio=2), Layout(name="right", ratio=1))
    layout["left"].split_column(Layout(name="trajectory", size=6), Layout(name="recent", ratio=1))
    layout["right"].split_column(Layout(name="summary", ratio=1), Layout(name="context", ratio=1))

    done = step >= len(state["prices"]) - 1
    layout["header"].update(build_header(state, step, done))
    layout["trajectory"].update(build_trajectory_panel(state, step, spark_width))
    layout["recent"].update(Panel(build_recent_steps_table(state, step, history_window), title="Recent Steps", box=box.ROUNDED))
    layout["summary"].update(Panel(build_summary_table(state, step), title="Step Summary", box=box.ROUNDED))
    layout["context"].update(Panel(build_context_table(state), title="Checkpoint Context", box=box.ROUNDED))
    layout["footer"].update(build_footer_panel(state, step))
    return layout


def main() -> None:
    args = parse_args()
    checkpoint_path = resolve_checkpoint_path(args)
    model, metadata = load_checkpoint(checkpoint_path, device=args.device)
    bundle, resolved_config = build_dataset(args, metadata)
    resolved_config["device"] = args.device
    state = build_inference_state(
        model=model,
        metadata=metadata,
        bundle=bundle,
        resolved_config=resolved_config,
        path_index=args.path_index,
        device=args.device,
    )

    total_steps = len(state["prices"])
    last_step = total_steps - 1 if args.max_steps <= 0 else min(total_steps, args.max_steps) - 1
    console = Console()

    try:
        with Live(
            render_dashboard(state, 0, args.history_window, args.spark_width),
            console=console,
            refresh_per_second=max(4, int(1.0 / max(args.delay, 0.05))),
            screen=True,
        ) as live:
            for step in range(last_step + 1):
                live.update(render_dashboard(state, step, args.history_window, args.spark_width), refresh=True)
                if step < last_step and args.delay > 0:
                    time.sleep(args.delay)
    except KeyboardInterrupt:
        console.print("\nInference TUI interrupted by user.")


if __name__ == "__main__":
    main()
