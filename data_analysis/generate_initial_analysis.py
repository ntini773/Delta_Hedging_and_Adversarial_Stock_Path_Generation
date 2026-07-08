from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm


DATA_FILE = ROOT / "stuff" / "20260205_option_minute_prices_expiry.csv"
OUTPUT_DIR = ROOT / "data_analysis"
EXPIRY_TIME = pd.Timestamp("2026-02-05 15:30:00")
SNAPSHOT_MINUTE = 110000
RISK_FREE_RATE = 0.05


def parse_option(symbol: str) -> tuple[float | None, str | None]:
    if symbol.endswith("CE"):
        return float(symbol[-7:-2]), "call"
    if symbol.endswith("PE"):
        return float(symbol[-7:-2]), "put"
    return None, None


def black_scholes(spot: float, strike: float, time_to_expiry: float, rate: float, vol: float, option_type: str):
    if time_to_expiry <= 0:
        payoff = max(0.0, spot - strike) if option_type == "call" else max(0.0, strike - spot)
        delta = 1.0 if option_type == "call" and spot > strike else 0.0
        delta = -1.0 if option_type == "put" and spot < strike else delta
        return payoff, delta, 0.0, 0.0, 0.0

    vol = max(vol, 1e-8)
    sqrt_t = np.sqrt(time_to_expiry)
    d1 = (np.log(spot / strike) + (rate + 0.5 * vol**2) * time_to_expiry) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t

    if option_type == "call":
        price = spot * norm.cdf(d1) - strike * np.exp(-rate * time_to_expiry) * norm.cdf(d2)
        delta = norm.cdf(d1)
        theta = -(spot * norm.pdf(d1) * vol) / (2 * sqrt_t) - rate * strike * np.exp(-rate * time_to_expiry) * norm.cdf(d2)
    else:
        price = strike * np.exp(-rate * time_to_expiry) * norm.cdf(-d2) - spot * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
        theta = -(spot * norm.pdf(d1) * vol) / (2 * sqrt_t) + rate * strike * np.exp(-rate * time_to_expiry) * norm.cdf(-d2)

    gamma = norm.pdf(d1) / (spot * vol * sqrt_t)
    vega = spot * norm.pdf(d1) * sqrt_t
    return price, delta, gamma, theta, vega


def solve_implied_volatility(price: float, spot: float, strike: float, time_to_expiry: float, rate: float, option_type: str) -> float:
    if price <= 0 or time_to_expiry <= 0:
        return np.nan

    low, high = 0.001, 5.0
    for _ in range(100):
        mid = (low + high) / 2
        model_price, _, _, _, _ = black_scholes(spot, strike, time_to_expiry, rate, mid, option_type)
        if abs(model_price - price) < 1e-5:
            return mid
        if model_price < price:
            low = mid
        else:
            high = mid
    return mid


def plot_greek(snapshot: pd.DataFrame, greek: str, title: str, output_name: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for option_type, color in [("call", "#1f77b4"), ("put", "#d62728")]:
        subset = snapshot[snapshot["option_type"] == option_type].sort_values("strike")
        ax.plot(subset["strike"], subset[greek], marker="o", linewidth=2, color=color, label=option_type)
    # If a dotted reference line is added here later, give it an explicit label.
    ax.set_title(title)
    ax.set_xlabel("Strike")
    ax.set_ylabel(greek.capitalize())
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / output_name, dpi=160)
    plt.close(fig)


def plot_intraday_future(futures: pd.DataFrame, chosen_date: int) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(futures["observation_time"], futures["future_price"], linewidth=2, color="#2f4b7c", label="future")
    ax.set_title(f"Intraday Future Price ({chosen_date})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Future Price (INR)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "future_intraday.png", dpi=160)
    plt.close(fig)


def plot_intraday_options(option_timeseries: pd.DataFrame, chosen_date: int, strike: float) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for option_type, color in [("call", "#1f77b4"), ("put", "#d62728")]:
        subset = option_timeseries[option_timeseries["option_type"] == option_type]
        ax.plot(subset["observation_time"], subset["option_price"], linewidth=2, color=color, label=option_type)
    ax.set_title(f"Intraday Option Prices for Strike {int(strike)} ({chosen_date})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Option Price (INR)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"options_intraday_strike_{int(strike)}.png", dpi=160)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(DATA_FILE)
    chosen_date = int(df["date"].sample(random_state=42).iloc[0])
    df = df[df["date"] == chosen_date].copy()
    df["observation_time"] = pd.to_datetime(
        df["date"].astype(str) + df["minute_end"].astype(str).str.zfill(6),
        format="%Y%m%d%H%M%S",
    )
    snapshot = df[(df["date"] == chosen_date) & (df["minute_end"] == SNAPSHOT_MINUTE)].copy()

    future_row = snapshot[snapshot["symbol"].str.contains("FUT")]
    spot = future_row["last_trade_price"].iloc[0] / 100.0

    options = snapshot[~snapshot["symbol"].str.contains("FUT")].copy()
    options[["strike", "option_type"]] = options["symbol"].apply(lambda symbol: pd.Series(parse_option(symbol)))
    options = options.dropna(subset=["strike", "option_type"]).copy()
    options["option_price"] = options["last_trade_price"] / 100.0

    snapshot_time = pd.to_datetime(f"{chosen_date} {SNAPSHOT_MINUTE:06d}", format="%Y%m%d %H%M%S")
    options["time_to_expiry"] = (EXPIRY_TIME - snapshot_time).total_seconds() / (365.25 * 24 * 3600)

    options["implied_volatility"] = options.apply(
        lambda row: solve_implied_volatility(
            row["option_price"], spot, row["strike"], row["time_to_expiry"], RISK_FREE_RATE, row["option_type"]
        ),
        axis=1,
    )

    greeks = options.apply(
        lambda row: pd.Series(
            black_scholes(
                spot, row["strike"], row["time_to_expiry"], RISK_FREE_RATE, row["implied_volatility"], row["option_type"]
            )[1:5],
            index=["delta", "gamma", "theta", "vega"],
        ),
        axis=1,
    )
    options = pd.concat([options, greeks], axis=1)

    sample_columns = ["symbol", "strike", "option_type", "option_price", "implied_volatility", "delta", "gamma", "theta", "vega"]
    sample_table = options[sample_columns].sort_values(["strike", "option_type"]).head(8)

    print(f"Chosen file: {DATA_FILE.name}")
    print(f"Chosen date: {chosen_date}")
    print(f"Expiry assumed: {EXPIRY_TIME}")
    print(f"Snapshot time: {snapshot_time}")
    print(f"Spot S in rupees: {spot:.2f}")
    print("Parsed strikes sample:", options["strike"].sort_values().head(6).tolist())
    print()
    print(sample_table.to_string(index=False))

    plot_greek(options, "delta", f"Delta vs Strike at {snapshot_time} (S={spot:.2f})", "delta_vs_strike.png")
    plot_greek(options, "gamma", f"Gamma vs Strike at {snapshot_time} (S={spot:.2f})", "gamma_vs_strike.png")
    plot_greek(options, "theta", f"Theta vs Strike at {snapshot_time} (S={spot:.2f})", "theta_vs_strike.png")
    plot_greek(options, "vega", f"Vega vs Strike at {snapshot_time} (S={spot:.2f})", "vega_vs_strike.png")

    iv_plot = options.sort_values("strike")
    fig, ax = plt.subplots(figsize=(10, 5))
    for option_type, color in [("call", "#1f77b4"), ("put", "#d62728")]:
        subset = iv_plot[iv_plot["option_type"] == option_type]
        ax.plot(subset["strike"], subset["implied_volatility"], marker="o", linewidth=2, color=color, label=option_type)
    # If a dotted reference line is added here later, give it an explicit label.
    ax.set_title(f"IV Smile at {snapshot_time} (S={spot:.2f})")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Implied Volatility")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "iv_smile.png", dpi=160)
    plt.close(fig)

    selected_strike = float(options.assign(distance_to_spot=(options["strike"] - spot).abs()).sort_values("distance_to_spot")["strike"].iloc[0])
    futures = df[df["symbol"].str.contains("FUT")].copy()
    futures["future_price"] = futures["last_trade_price"] / 100.0
    option_timeseries = df[~df["symbol"].str.contains("FUT")].copy()
    option_timeseries[["strike", "option_type"]] = option_timeseries["symbol"].apply(lambda symbol: pd.Series(parse_option(symbol)))
    option_timeseries = option_timeseries[option_timeseries["strike"] == selected_strike].copy()
    option_timeseries["option_price"] = option_timeseries["last_trade_price"] / 100.0

    plot_intraday_future(futures, chosen_date)
    plot_intraday_options(option_timeseries, chosen_date, selected_strike)

    readme = "\n".join(
        [
            "# Initial Data Analysis",
            "",
            f"- File used: `{DATA_FILE.name}`",
            f"- Date used: `{chosen_date}`",
            f"- Snapshot minute: `{SNAPSHOT_MINUTE}`",
            f"- Expiry assumed: `{EXPIRY_TIME}`",
            f"- Spot S after paisa conversion: `{spot:.2f}`",
            f"- Selected intraday option strike: `{int(selected_strike)}`",
            "",
            "Generated plots:",
            "- `iv_smile.png`",
            "- `delta_vs_strike.png`",
            "- `gamma_vs_strike.png`",
            "- `theta_vs_strike.png`",
            "- `vega_vs_strike.png`",
            "- `future_intraday.png`",
            f"- `options_intraday_strike_{int(selected_strike)}.png`",
        ]
    )
    (OUTPUT_DIR / "README.md").write_text(readme)


if __name__ == "__main__":
    main()
