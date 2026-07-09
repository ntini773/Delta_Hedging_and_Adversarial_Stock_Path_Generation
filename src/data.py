from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.black_scholes import black_scholes_price_and_greeks, solve_implied_volatility

JUMP_DIFFUSION_PARAMS = {
    "jump_intensity": 120.0,
    "jump_mean": -0.03,
    "jump_std": 0.12,
}


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
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    shocks = torch.randn((num_paths, num_steps), generator=generator, dtype=torch.float32)
    log_returns = (drift - 0.5 * volatility * volatility) * dt + volatility * math.sqrt(dt) * shocks
    cumulative_log_returns = torch.cumsum(log_returns, dim=1)
    paths = torch.empty((num_paths, num_steps + 1), dtype=torch.float32)
    paths[:, 0] = float(start_price)
    paths[:, 1:] = float(start_price) * torch.exp(cumulative_log_returns)
    return paths.numpy()


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
    jump_drift = jump_intensity * (math.exp(jump_mean + 0.5 * jump_std * jump_std) - 1.0)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    diffusion_shocks = torch.randn((num_paths, num_steps), generator=generator, dtype=torch.float32)
    jump_counts = torch.poisson(
        torch.full((num_paths, num_steps), fill_value=jump_intensity * dt, dtype=torch.float32),
        generator=generator,
    )
    jump_noise = torch.randn((num_paths, num_steps), generator=generator, dtype=torch.float32)
    jump_log_sizes = torch.where(
        jump_counts > 0,
        jump_counts * jump_mean + torch.sqrt(jump_counts) * jump_std * jump_noise,
        torch.zeros_like(jump_counts),
    )
    log_returns = (
        (drift - jump_drift - 0.5 * volatility * volatility) * dt
        + volatility * math.sqrt(dt) * diffusion_shocks
        + jump_log_sizes
    )
    cumulative_log_returns = torch.cumsum(log_returns, dim=1)
    paths = torch.empty((num_paths, num_steps + 1), dtype=torch.float32)
    paths[:, 0] = float(start_price)
    paths[:, 1:] = float(start_price) * torch.exp(cumulative_log_returns)
    return paths.numpy()


def compute_black_scholes_reference(
    paths: np.ndarray,
    strike: float,
    rate: float,
    volatility: float,
    option_type: str,
    total_horizon_years: float,
) -> dict[str, np.ndarray]:
    num_time_points = paths.shape[1]
    if num_time_points <= 1:
        raise ValueError("Paths must contain at least two time points.")

    dt = total_horizon_years / (num_time_points - 1)
    time_points = np.maximum(total_horizon_years - np.arange(num_time_points, dtype=np.float32) * dt, 0.0)
    time_grid = np.broadcast_to(time_points, paths.shape).copy()

    # Black-Scholes is the main preprocessing bottleneck, so compute it over the full
    # path matrix at once instead of looping path-by-path in Python.
    with torch.no_grad():
        spot = torch.tensor(paths, dtype=torch.float32)
        time_to_expiry = torch.tensor(time_grid, dtype=torch.float32)
        strike_tensor = torch.full_like(spot, fill_value=float(strike))
        sigma_tensor = torch.full_like(spot, fill_value=float(max(volatility, 1e-8)))
        rate_tensor = torch.full_like(spot, fill_value=float(rate))

        safe_time = torch.clamp(time_to_expiry, min=1e-12)
        sqrt_t = torch.sqrt(safe_time)
        d1 = (torch.log(spot / strike_tensor) + (rate_tensor + 0.5 * sigma_tensor * sigma_tensor) * safe_time) / (
            sigma_tensor * sqrt_t
        )
        d2 = d1 - sigma_tensor * sqrt_t

        normal_cdf_d1 = 0.5 * (1.0 + torch.erf(d1 / math.sqrt(2.0)))
        normal_cdf_d2 = 0.5 * (1.0 + torch.erf(d2 / math.sqrt(2.0)))
        normal_cdf_neg_d1 = 0.5 * (1.0 + torch.erf(-d1 / math.sqrt(2.0)))
        normal_cdf_neg_d2 = 0.5 * (1.0 + torch.erf(-d2 / math.sqrt(2.0)))
        normal_pdf_d1 = torch.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
        discount = torch.exp(-rate_tensor * safe_time)

        if option_type == "call":
            bs_prices = spot * normal_cdf_d1 - strike_tensor * discount * normal_cdf_d2
            bs_deltas = normal_cdf_d1
            bs_thetas = -(spot * normal_pdf_d1 * sigma_tensor) / (2.0 * sqrt_t) - rate_tensor * strike_tensor * discount * normal_cdf_d2
        else:
            bs_prices = strike_tensor * discount * normal_cdf_neg_d2 - spot * normal_cdf_neg_d1
            bs_deltas = normal_cdf_d1 - 1.0
            bs_thetas = -(spot * normal_pdf_d1 * sigma_tensor) / (2.0 * sqrt_t) + rate_tensor * strike_tensor * discount * normal_cdf_neg_d2

        bs_gammas = normal_pdf_d1 / (spot * sigma_tensor * sqrt_t)
        bs_vegas = spot * normal_pdf_d1 * sqrt_t

        expired_mask = time_to_expiry <= 0.0
        if option_type == "call":
            expired_payoff = torch.clamp(spot - strike_tensor, min=0.0)
            expired_delta = torch.where(spot > strike_tensor, 1.0, 0.0)
        else:
            expired_payoff = torch.clamp(strike_tensor - spot, min=0.0)
            expired_delta = torch.where(spot < strike_tensor, -1.0, 0.0)

        bs_prices = torch.where(expired_mask, expired_payoff, bs_prices)
        bs_deltas = torch.where(expired_mask, expired_delta, bs_deltas)
        bs_gammas = torch.where(expired_mask, torch.zeros_like(bs_gammas), bs_gammas)
        bs_thetas = torch.where(expired_mask, torch.zeros_like(bs_thetas), bs_thetas)
        bs_vegas = torch.where(expired_mask, torch.zeros_like(bs_vegas), bs_vegas)

    return {
        "bs_prices": bs_prices.cpu().numpy().astype(np.float32),
        "bs_deltas": bs_deltas.cpu().numpy().astype(np.float32),
        "bs_gammas": bs_gammas.cpu().numpy().astype(np.float32),
        "bs_thetas": bs_thetas.cpu().numpy().astype(np.float32),
        "bs_vegas": bs_vegas.cpu().numpy().astype(np.float32),
        "time_grid": time_grid.astype(np.float32),
        "implied_volatility": np.full_like(paths, fill_value=volatility, dtype=np.float32),
    }


def split_train_validation_test(
    paths: np.ndarray,
    references: dict[str, np.ndarray],
    test_ratio: float,
    validation_ratio: float,
    seed: int,
) -> dict:
    if test_ratio <= 0 or validation_ratio <= 0 or test_ratio + validation_ratio >= 1.0:
        raise ValueError("test_ratio and validation_ratio must be positive and sum to less than 1.")

    rng = np.random.default_rng(seed)
    indices = np.arange(paths.shape[0])
    rng.shuffle(indices)
    train_end = int(paths.shape[0] * (1.0 - test_ratio - validation_ratio))
    validation_end = int(paths.shape[0] * (1.0 - test_ratio))
    train_idx = indices[:train_end]
    validation_idx = indices[train_end:validation_end]
    test_idx = indices[validation_end:]
    return {
        "train": {
            "paths": paths[train_idx],
            **{name: values[train_idx] for name, values in references.items()},
        },
        "validation": {
            "paths": paths[validation_idx],
            **{name: values[validation_idx] for name, values in references.items()},
        },
        "test": {
            "paths": paths[test_idx],
            **{name: values[test_idx] for name, values in references.items()},
        },
        "train_indices": train_idx.tolist(),
        "validation_indices": validation_idx.tolist(),
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
    jump_intensity: float,
    jump_mean: float,
    jump_std: float,
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
            jump_intensity=jump_intensity,
            jump_mean=jump_mean,
            jump_std=jump_std,
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
    validation_ratio: float,
    jump_intensity: float = JUMP_DIFFUSION_PARAMS["jump_intensity"],
    jump_mean: float = JUMP_DIFFUSION_PARAMS["jump_mean"],
    jump_std: float = JUMP_DIFFUSION_PARAMS["jump_std"],
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
        jump_intensity=jump_intensity,
        jump_mean=jump_mean,
        jump_std=jump_std,
    )
    references = compute_black_scholes_reference(
        paths=paths,
        strike=context["strike"],
        rate=rate,
        volatility=volatility,
        option_type=context["option_type"],
        total_horizon_years=context["time_horizon_years"],
    )
    split = split_train_validation_test(
        paths=paths,
        references=references,
        test_ratio=test_ratio,
        validation_ratio=validation_ratio,
        seed=seed,
    )
    config = {
        "csv_path": str(csv_path),
        "expiry_time": expiry_time,
        "regime": regime,
        "num_paths": num_paths,
        "volatility": volatility,
        "rate": rate,
        "seed": seed,
        "test_ratio": test_ratio,
        "validation_ratio": validation_ratio,
        "target_option_symbol": context["target_option_symbol"],
    }
    if regime == "jump_diffusion":
        config.update(
            {
                "jump_intensity": jump_intensity,
                "jump_mean": jump_mean,
                "jump_std": jump_std,
            }
        )

    return {
        "context": context,
        "config": config,
        "train": split["train"],
        "validation": split["validation"],
        "test": split["test"],
        "train_indices": split["train_indices"],
        "validation_indices": split["validation_indices"],
        "test_indices": split["test_indices"],
    }


def to_torch(array: np.ndarray, device: str) -> torch.Tensor:
    return torch.tensor(array, dtype=torch.float32, device=device)


def build_v1_features(
    price_paths: torch.Tensor,
    bs_deltas: torch.Tensor,
    bs_gammas: torch.Tensor,
    bs_thetas: torch.Tensor,
    bs_vegas: torch.Tensor,
    time_to_expiry: torch.Tensor,
    implied_volatility: torch.Tensor,
    strike: float,
    previous_delta: torch.Tensor,
    running_pnl: torch.Tensor,
    step: int,
) -> torch.Tensor:
    spot = price_paths[:, step : step + 1]
    bs_delta = bs_deltas[:, step : step + 1]
    return torch.cat([spot, bs_delta, previous_delta], dim=1)


def build_v2_features(
    price_paths: torch.Tensor,
    bs_deltas: torch.Tensor,
    bs_gammas: torch.Tensor,
    bs_thetas: torch.Tensor,
    bs_vegas: torch.Tensor,
    time_to_expiry: torch.Tensor,
    implied_volatility: torch.Tensor,
    strike: float,
    previous_delta: torch.Tensor,
    running_pnl: torch.Tensor,
    step: int,
) -> torch.Tensor:
    spot = price_paths[:, step : step + 1]
    log_moneyness = torch.log(spot / strike)
    remaining_time = time_to_expiry[:, step : step + 1]
    bs_delta = bs_deltas[:, step : step + 1]
    iv = implied_volatility[:, step : step + 1]
    return torch.cat([log_moneyness, remaining_time, bs_delta, iv, previous_delta], dim=1)


def calculate_realized_volatility(price_paths: torch.Tensor, step: int, window_size: int = 10) -> torch.Tensor:
    if step == 0:
        return torch.zeros((price_paths.shape[0], 1), device=price_paths.device, dtype=price_paths.dtype)
    start = max(0, step - window_size)
    history = price_paths[:, start : step + 1]
    log_returns = torch.log(history[:, 1:] / history[:, :-1])
    if log_returns.shape[1] == 0:
        return torch.zeros((price_paths.shape[0], 1), device=price_paths.device, dtype=price_paths.dtype)
    return log_returns.std(dim=1, unbiased=False, keepdim=True)


def build_v3_features(
    price_paths: torch.Tensor,
    bs_deltas: torch.Tensor,
    bs_gammas: torch.Tensor,
    bs_thetas: torch.Tensor,
    bs_vegas: torch.Tensor,
    time_to_expiry: torch.Tensor,
    implied_volatility: torch.Tensor,
    strike: float,
    previous_delta: torch.Tensor,
    running_pnl: torch.Tensor,
    step: int,
) -> torch.Tensor:
    spot = price_paths[:, step : step + 1]
    log_moneyness = torch.log(spot / strike)
    remaining_time = time_to_expiry[:, step : step + 1]
    bs_delta = bs_deltas[:, step : step + 1]
    bs_gamma = bs_gammas[:, step : step + 1]
    bs_theta = bs_thetas[:, step : step + 1]
    bs_vega = bs_vegas[:, step : step + 1]
    iv = implied_volatility[:, step : step + 1]
    realized_volatility = calculate_realized_volatility(price_paths, step=step)
    step_fraction = torch.full_like(remaining_time, fill_value=step / max(price_paths.shape[1] - 1, 1))
    return torch.cat(
        [
            log_moneyness,
            remaining_time,
            bs_delta,
            bs_gamma,
            bs_theta,
            bs_vega,
            iv,
            realized_volatility,
            step_fraction,
            running_pnl,
            previous_delta,
        ],
        dim=1,
    )


def build_v4_features(
    price_paths: torch.Tensor,
    bs_deltas: torch.Tensor,
    bs_gammas: torch.Tensor,
    bs_thetas: torch.Tensor,
    bs_vegas: torch.Tensor,
    time_to_expiry: torch.Tensor,
    implied_volatility: torch.Tensor,
    strike: float,
    previous_delta: torch.Tensor,
    running_pnl: torch.Tensor,
    step: int,
) -> torch.Tensor:
    spot = price_paths[:, step : step + 1]
    remaining_time = time_to_expiry[:, step : step + 1]
    bs_delta = bs_deltas[:, step : step + 1]
    bs_gamma = bs_gammas[:, step : step + 1]
    bs_theta = bs_thetas[:, step : step + 1]
    bs_vega = bs_vegas[:, step : step + 1]
    iv = implied_volatility[:, step : step + 1]
    log_moneyness = torch.log(torch.clamp(spot / strike, min=1e-8))
    sqrt_time = torch.sqrt(torch.clamp(remaining_time, min=0.0))
    realized_volatility = calculate_realized_volatility(price_paths, step=step)
    step_fraction = torch.full_like(remaining_time, fill_value=step / max(price_paths.shape[1] - 1, 1))
    running_pnl_scaled = running_pnl / max(float(strike), 1e-8)
    delta_gap_to_prev = previous_delta - bs_delta
    scaled_bs_gamma = bs_gamma * spot
    scaled_bs_theta = bs_theta * torch.clamp(remaining_time, min=1e-6)
    scaled_bs_vega = bs_vega / max(float(strike), 1e-8)
    if step == 0:
        instant_log_return = torch.zeros_like(spot)
    else:
        previous_spot = price_paths[:, step - 1 : step]
        instant_log_return = torch.log(torch.clamp(spot / previous_spot, min=1e-8))

    return torch.cat(
        [
            log_moneyness,
            sqrt_time,
            bs_delta,
            delta_gap_to_prev,
            scaled_bs_gamma,
            scaled_bs_theta,
            scaled_bs_vega,
            iv,
            realized_volatility,
            instant_log_return,
            running_pnl_scaled,
            step_fraction,
            previous_delta,
        ],
        dim=1,
    )


FEATURE_BUILDERS = {
    "v1": build_v1_features,
    "v2": build_v2_features,
    "v3": build_v3_features,
    "v4": build_v4_features,
}
