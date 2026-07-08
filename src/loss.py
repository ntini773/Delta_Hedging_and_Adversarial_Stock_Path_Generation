from __future__ import annotations

import math

import torch


def calculate_transaction_costs(
    price_paths: torch.Tensor,
    hedge_paths: torch.Tensor,
    transaction_cost_rate: float,
) -> torch.Tensor:
    if transaction_cost_rate <= 0.0:
        return torch.zeros(price_paths.shape[0], device=price_paths.device, dtype=price_paths.dtype)

    initial_hedge = torch.zeros((hedge_paths.shape[0], 1), device=hedge_paths.device, dtype=hedge_paths.dtype)
    hedge_changes = torch.diff(hedge_paths, dim=1, prepend=initial_hedge)
    running_costs = transaction_cost_rate * price_paths * hedge_changes.abs()
    final_unwind = transaction_cost_rate * price_paths[:, -1] * hedge_paths[:, -1].abs()
    return running_costs.sum(dim=1) + final_unwind


def calculate_hedging_pnl(
    price_paths: torch.Tensor,
    hedge_paths: torch.Tensor,
    strike: float,
    option_type: str,
    transaction_cost_rate: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    price_changes = price_paths[:, 1:] - price_paths[:, :-1]
    trading_pnl = (hedge_paths[:, :-1] * price_changes).sum(dim=1)

    if option_type == "call":
        payoff = torch.clamp(price_paths[:, -1] - strike, min=0.0)
    else:
        payoff = torch.clamp(strike - price_paths[:, -1], min=0.0)

    transaction_costs = calculate_transaction_costs(price_paths, hedge_paths, transaction_cost_rate)
    pnl = -payoff + trading_pnl - transaction_costs
    return pnl, transaction_costs


def calculate_cvar_loss(pnl_values: torch.Tensor, alpha: float = 0.05) -> torch.Tensor:
    sorted_pnl = torch.sort(pnl_values).values
    worst_count = max(1, int(math.floor(sorted_pnl.shape[0] * alpha)))
    worst_tail = sorted_pnl[:worst_count]
    return -worst_tail.mean()


def calculate_var(pnl_values: torch.Tensor, alpha: float = 0.05) -> float:
    sorted_pnl = torch.sort(pnl_values).values
    index = max(0, int(math.floor(sorted_pnl.shape[0] * alpha)) - 1)
    return float(sorted_pnl[index].item())


def calculate_cvar(pnl_values: torch.Tensor, alpha: float = 0.05) -> float:
    return float(calculate_cvar_loss(pnl_values, alpha=alpha).item())


def calculate_downside_deviation(pnl_values: torch.Tensor) -> float:
    downside = torch.minimum(pnl_values, torch.zeros_like(pnl_values))
    return float(torch.sqrt(torch.mean(downside * downside)).item())
