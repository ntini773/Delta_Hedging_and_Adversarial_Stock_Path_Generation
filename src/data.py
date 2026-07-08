from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.black_scholes import black_scholes_price_and_greeks, solve_implied_volatility


def parse_option_symbol(symbol: str) -> tuple[float | None, str | None]:
    if symbol.endswith("CE"):
        return float(symbol[-7:-2]), "call"
    if symbol.endswith("PE"):
        return float(symbol[-7:-2]), "put"
    return None, None


def load_option_csv(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["observation_time"] = pd.to_datetime(
        df["date"].astype(str) + df["minute_end"].astype(str).str.zfill(6),
        format="%Y%m%d%H%M%S",
    )
    return df


def build_real_market_context(
    csv_path: str | Path,
    expiry_time: str,
    snapshot_minute: int = 110000,
    target_option_symbol: str | None = None,
) -> dict:
    df = load_option_csv(csv_path)
    chosen_date = int(df["date"].iloc[0])
    snapshot = df[df["minute_end"] == snapshot_minute].copy()

    future_row = snapshot[snapshot["symbol"].str.contains("FUT")]
    spot = float(future_row["last_trade_price"].iloc[0] / 100.0)
    snapshot_time = pd.to_datetime(f"{chosen_date} {snapshot_minute:06d}", format="%Y%m%d %H%M%S")
    expiry_timestamp = pd.Timestamp(expiry_time)

    options = snapshot[~snapshot["symbol"].str.contains("FUT")].copy()
    options[["strike", "option_type"]] = options["symbol"].apply(lambda value: pd.Series(parse_option_symbol(value)))
    options = options.dropna(subset=["strike", "option_type"]).copy()
    options["option_price"] = options["last_trade_price"] / 100.0

    if target_option_symbol is None:
        calls = options[options["option_type"] == "call"].copy()
        calls["distance_to_spot"] = (calls["strike"] - spot).abs()
        target_option_symbol = str(calls.sort_values("distance_to_spot")["symbol"].iloc[0])

    target_row = options[options["symbol"] == target_option_symbol].iloc[0]
    strike = float(target_row["strike"])
    option_type = str(target_row["option_type"])
    observed_price = float(target_row["option_price"])
    time_to_expiry = (expiry_timestamp - snapshot_time).total_seconds() / (365.25 * 24 * 3600)
    implied_volatility = solve_implied_volatility(
        observed_price=observed_price,
        spot=spot,
        strike=strike,
        time_to_expiry=time_to_expiry,
        rate=0.05,
        option_type=option_type,
    )

    future_series = df[df["symbol"].str.contains("FUT")][["observation_time", "last_trade_price"]].copy()
    future_series["future_price"] = future_series["last_trade_price"] / 100.0
    option_series = df[df["symbol"] == target_option_symbol][["observation_time", "last_trade_price"]].copy()
    option_series["option_price"] = option_series["last_trade_price"] / 100.0
    merged = future_series.merge(option_series, on="observation_time", suffixes=("_fut", "_opt")).sort_values(
        "observation_time"
    )

    return {
        "chosen_date": chosen_date,
        "snapshot_time": snapshot_time,
        "expiry_time": expiry_timestamp,
        "spot": spot,
        "strike": strike,
        "option_type": option_type,
        "target_option_symbol": target_option_symbol,
        "initial_implied_volatility": float(implied_volatility),
        "num_time_steps": len(merged) - 1,
        "time_horizon_years": (
            merged["observation_time"].iloc[-1] - merged["observation_time"].iloc[0]
        ).total_seconds()
        / (365.25 * 24 * 3600),
    }


def generate_gbm_paths(
    start_price: float,
    num_steps: int,
    dt: float,
    num_paths: int,
    drift: float,
    volatility: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    paths = np.zeros((num_paths, num_steps + 1), dtype=np.float32)
    paths[:, 0] = start_price
    for step in range(1, num_steps + 1):
        shocks = rng.standard_normal(num_paths)
        paths[:, step] = paths[:, step - 1] * np.exp(
            (drift - 0.5 * volatility * volatility) * dt + volatility * math.sqrt(dt) * shocks
        )
    return paths


def generate_jump_diffusion_paths(
    start_price: float,
    num_steps: int,
    dt: float,
    num_paths: int,
    drift: float,
    volatility: float,
    jump_intensity: float,
    jump_mean: float,
    jump_std: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    paths = np.zeros((num_paths, num_steps + 1), dtype=np.float32)
    paths[:, 0] = start_price
    jump_drift = jump_intensity * (math.exp(jump_mean + 0.5 * jump_std * jump_std) - 1.0)
    for step in range(1, num_steps + 1):
        diffusion_shocks = rng.standard_normal(num_paths)
        jump_counts = rng.poisson(jump_intensity * dt, size=num_paths)
        jump_sizes = np.where(
            jump_counts > 0,
            np.exp(jump_counts * jump_mean + np.sqrt(jump_counts) * jump_std * rng.standard_normal(num_paths)),
            1.0,
        )
        paths[:, step] = paths[:, step - 1] * np.exp(
            (drift - jump_drift - 0.5 * volatility * volatility) * dt + volatility * math.sqrt(dt) * diffusion_shocks
        ) * jump_sizes
    return paths


def compute_black_scholes_reference(
    paths: np.ndarray,
    strike: float,
    rate: float,
    volatility: float,
    option_type: str,
    total_horizon_years: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_time_points = paths.shape[1]
    if num_time_points <= 1:
        raise ValueError("Paths must contain at least two time points.")

    dt = total_horizon_years / (num_time_points - 1)
    bs_prices = np.zeros_like(paths, dtype=np.float32)
    bs_deltas = np.zeros_like(paths, dtype=np.float32)
    time_grid = np.zeros_like(paths, dtype=np.float32)

    for step in range(num_time_points):
        remaining_time = max(total_horizon_years - step * dt, 0.0)
        time_grid[:, step] = remaining_time
        for path_index in range(paths.shape[0]):
            greeks = black_scholes_price_and_greeks(
                spot=float(paths[path_index, step]),
                strike=strike,
                time_to_expiry=remaining_time,
                rate=rate,
                volatility=volatility,
                option_type=option_type,
            )
            bs_prices[path_index, step] = greeks["price"]
            bs_deltas[path_index, step] = greeks["delta"]
    return bs_prices, bs_deltas, time_grid


def split_train_test(
    paths: np.ndarray,
    bs_prices: np.ndarray,
    bs_deltas: np.ndarray,
    time_grid: np.ndarray,
    test_ratio: float,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    indices = np.arange(paths.shape[0])
    rng.shuffle(indices)
    split_index = int(paths.shape[0] * (1.0 - test_ratio))
    train_idx = indices[:split_index]
    test_idx = indices[split_index:]
    return {
        "train": {
            "paths": paths[train_idx],
            "bs_prices": bs_prices[train_idx],
            "bs_deltas": bs_deltas[train_idx],
            "time_grid": time_grid[train_idx],
        },
        "test": {
            "paths": paths[test_idx],
            "bs_prices": bs_prices[test_idx],
            "bs_deltas": bs_deltas[test_idx],
            "time_grid": time_grid[test_idx],
        },
        "test_indices": test_idx.tolist(),
    }


def generate_regime_dataset(
    regime: str,
    start_price: float,
    num_steps: int,
    total_horizon_years: float,
    num_paths: int,
    rate: float,
    volatility: float,
    seed: int,
) -> np.ndarray:
    dt = total_horizon_years / num_steps
    if regime == "gbm":
        return generate_gbm_paths(
            start_price=start_price,
            num_steps=num_steps,
            dt=dt,
            num_paths=num_paths,
            drift=rate,
            volatility=volatility,
            seed=seed,
        )
    if regime == "jump_diffusion":
        return generate_jump_diffusion_paths(
            start_price=start_price,
            num_steps=num_steps,
            dt=dt,
            num_paths=num_paths,
            drift=rate,
            volatility=volatility,
            jump_intensity=25.0,
            jump_mean=-0.02,
            jump_std=0.08,
            seed=seed,
        )
    raise ValueError(f"Unsupported regime: {regime}")


def prepare_dataset_bundle(
    csv_path: str | Path,
    expiry_time: str,
    regime: str,
    num_paths: int,
    volatility: float,
    rate: float,
    seed: int,
    test_ratio: float,
    target_option_symbol: str | None = None,
) -> dict:
    context = build_real_market_context(
        csv_path=csv_path,
        expiry_time=expiry_time,
        target_option_symbol=target_option_symbol,
    )
    paths = generate_regime_dataset(
        regime=regime,
        start_price=context["spot"],
        num_steps=context["num_time_steps"],
        total_horizon_years=context["time_horizon_years"],
        num_paths=num_paths,
        rate=rate,
        volatility=volatility,
        seed=seed,
    )
    bs_prices, bs_deltas, time_grid = compute_black_scholes_reference(
        paths=paths,
        strike=context["strike"],
        rate=rate,
        volatility=volatility,
        option_type=context["option_type"],
        total_horizon_years=context["time_horizon_years"],
    )
    split = split_train_test(
        paths=paths,
        bs_prices=bs_prices,
        bs_deltas=bs_deltas,
        time_grid=time_grid,
        test_ratio=test_ratio,
        seed=seed,
    )
    return {
        "context": context,
        "config": {
            "csv_path": str(csv_path),
            "expiry_time": expiry_time,
            "regime": regime,
            "num_paths": num_paths,
            "volatility": volatility,
            "rate": rate,
            "seed": seed,
            "test_ratio": test_ratio,
            "target_option_symbol": context["target_option_symbol"],
        },
        "train": split["train"],
        "test": split["test"],
        "test_indices": split["test_indices"],
    }


def to_torch(array: np.ndarray, device: str) -> torch.Tensor:
    return torch.tensor(array, dtype=torch.float32, device=device)


def build_v1_features(
    price_paths: torch.Tensor,
    bs_deltas: torch.Tensor,
    time_to_expiry: torch.Tensor,
    implied_volatility: torch.Tensor,
    strike: float,
    previous_delta: torch.Tensor,
    step: int,
) -> torch.Tensor:
    spot = price_paths[:, step : step + 1]
    bs_delta = bs_deltas[:, step : step + 1]
    return torch.cat([spot, bs_delta, previous_delta], dim=1)


def build_v2_features(
    price_paths: torch.Tensor,
    bs_deltas: torch.Tensor,
    time_to_expiry: torch.Tensor,
    implied_volatility: torch.Tensor,
    strike: float,
    previous_delta: torch.Tensor,
    step: int,
) -> torch.Tensor:
    spot = price_paths[:, step : step + 1]
    log_moneyness = torch.log(spot / strike)
    remaining_time = time_to_expiry[:, step : step + 1]
    bs_delta = bs_deltas[:, step : step + 1]
    iv = implied_volatility[:, step : step + 1]
    return torch.cat([log_moneyness, remaining_time, bs_delta, iv, previous_delta], dim=1)


FEATURE_BUILDERS = {
    "v1": build_v1_features,
    "v2": build_v2_features,
}
