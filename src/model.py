from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelSpec:
    name: str
    version: str
    features: list[str]
    hidden_dims: list[int]
    transaction_cost_rate: float
    default_regime: str
    changes_from_previous: list[str]
    alignment_weight: float = 0.0
    turnover_weight: float = 0.0


MODEL_SPECS = {
    "v1": ModelSpec(
        name="deep_hedger_v1",
        version="v1",
        features=["spot", "bs_delta", "prev_delta"],
        hidden_dims=[32],
        transaction_cost_rate=0.001,
        default_regime="gbm",
        changes_from_previous=[
            "Matches the notebook feature set: spot, Black-Scholes delta, previous hedge.",
            "Uses a single hidden layer with tanh activations.",
            "Optimizes terminal CVaR with the same transaction-cost model used by the benchmark.",
        ],
    ),
    "v2": ModelSpec(
        name="deep_hedger_v2",
        version="v2",
        features=["log_moneyness", "time_to_expiry", "bs_delta", "implied_volatility", "prev_delta"],
        hidden_dims=[64, 64],
        transaction_cost_rate=0.001,
        default_regime="gbm",
        changes_from_previous=[
            "Adds normalized state features instead of raw spot only.",
            "Adds time-to-expiry and implied-volatility context to the hedge decision.",
            "Uses a two-layer MLP and includes transaction costs in training and evaluation.",
        ],
    ),
    "v3": ModelSpec(
        name="deep_hedger_v3",
        version="v3",
        features=[
            "log_moneyness",
            "time_to_expiry",
            "bs_delta",
            "bs_gamma",
            "bs_theta",
            "bs_vega",
            "implied_volatility",
            "realized_volatility",
            "step_fraction",
            "running_pnl",
            "prev_delta",
        ],
        hidden_dims=[128, 128, 64],
        transaction_cost_rate=0.001,
        default_regime="jump_diffusion",
        changes_from_previous=[
            "Adds path-state features such as realized volatility, running P&L, and normalized step index.",
            "Uses a deeper three-layer MLP for a richer hedge policy without introducing recurrent layers.",
            "Intended to be trained on jump-diffusion paths while keeping the same benchmark interface.",
        ],
        turnover_weight=0.001,
    ),
    "v4": ModelSpec(
        name="deep_hedger_v4",
        version="v4",
        features=[
            "log_moneyness",
            "sqrt_time_to_expiry",
            "bs_delta",
            "delta_gap_to_prev",
            "scaled_bs_gamma",
            "scaled_bs_theta",
            "scaled_bs_vega",
            "implied_volatility",
            "realized_volatility",
            "instant_log_return",
            "running_pnl_scaled",
            "step_fraction",
            "prev_delta",
        ],
        hidden_dims=[256, 256, 128],
        transaction_cost_rate=0.001,
        default_regime="jump_diffusion",
        changes_from_previous=[
            "Predicts a residual adjustment on top of Black-Scholes delta instead of an unconstrained absolute hedge.",
            "Uses normalized market-state features so the policy is less likely to collapse to a flat hedge.",
            "Adds alignment and turnover regularization to keep training stable on large synthetic batches.",
        ],
        alignment_weight=0.05,
        turnover_weight=0.002,
    ),
}


class DeepHedgerMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.Tanh())
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        layers.append(nn.Tanh())
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class ResidualDeepHedgerMLP(nn.Module):
    uses_base_delta = True

    def __init__(self, input_dim: int, hidden_dims: list[int], residual_scale: float = 0.35):
        super().__init__()
        layers: list[nn.Module] = [nn.LayerNorm(input_dim)]
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.SiLU())
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        self.network = nn.Sequential(*layers)
        self.residual_scale = residual_scale

    def forward(self, features: torch.Tensor, base_delta: torch.Tensor) -> torch.Tensor:
        residual = self.residual_scale * torch.tanh(self.network(features))
        return torch.clamp(base_delta + residual, min=-1.0, max=1.0)


def build_model(model_version: str) -> nn.Module:
    spec = MODEL_SPECS[model_version]
    if model_version == "v4":
        return ResidualDeepHedgerMLP(input_dim=len(spec.features), hidden_dims=spec.hidden_dims)
    return DeepHedgerMLP(input_dim=len(spec.features), hidden_dims=spec.hidden_dims)


def build_bs_delta_policy() -> None:
    return None


def run_deep_hedger(
    model: nn.Module,
    feature_builder,
    price_paths: torch.Tensor,
    bs_deltas: torch.Tensor,
    bs_gammas: torch.Tensor,
    bs_thetas: torch.Tensor,
    bs_vegas: torch.Tensor,
    time_to_expiry: torch.Tensor,
    implied_volatility: torch.Tensor,
    strike: float,
) -> torch.Tensor:
    batch_size, num_time_points = price_paths.shape
    previous_delta = torch.zeros((batch_size, 1), device=price_paths.device, dtype=price_paths.dtype)
    running_pnl = torch.zeros((batch_size, 1), device=price_paths.device, dtype=price_paths.dtype)
    hedge_steps = []

    for step in range(num_time_points):
        current_bs_delta = bs_deltas[:, step : step + 1]
        features = feature_builder(
            price_paths=price_paths,
            bs_deltas=bs_deltas,
            bs_gammas=bs_gammas,
            bs_thetas=bs_thetas,
            bs_vegas=bs_vegas,
            time_to_expiry=time_to_expiry,
            implied_volatility=implied_volatility,
            strike=strike,
            previous_delta=previous_delta,
            running_pnl=running_pnl,
            step=step,
        )
        if getattr(model, "uses_base_delta", False):
            current_delta = model(features, base_delta=current_bs_delta)
        else:
            current_delta = model(features)
        hedge_steps.append(current_delta)
        if step < num_time_points - 1:
            price_change = price_paths[:, step + 1 : step + 2] - price_paths[:, step : step + 1]
            running_pnl = running_pnl + current_delta * price_change
        previous_delta = current_delta

    return torch.cat(hedge_steps, dim=1)
