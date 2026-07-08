from __future__ import annotations

import math


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def option_payoff(spot: float, strike: float, option_type: str) -> float:
    if option_type == "call":
        return max(0.0, spot - strike)
    return max(0.0, strike - spot)


def black_scholes_price_and_greeks(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    option_type: str,
) -> dict[str, float]:
    if time_to_expiry <= 0.0:
        payoff = option_payoff(spot, strike, option_type)
        if option_type == "call":
            delta = 1.0 if spot > strike else 0.0
        else:
            delta = -1.0 if spot < strike else 0.0
        return {
            "price": payoff,
            "delta": delta,
            "gamma": 0.0,
            "theta": 0.0,
            "vega": 0.0,
        }

    sigma = max(volatility, 1e-8)
    sqrt_t = math.sqrt(time_to_expiry)
    d1 = (math.log(spot / strike) + (rate + 0.5 * sigma * sigma) * time_to_expiry) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t

    if option_type == "call":
        price = spot * normal_cdf(d1) - strike * math.exp(-rate * time_to_expiry) * normal_cdf(d2)
        delta = normal_cdf(d1)
        theta = -(spot * normal_pdf(d1) * sigma) / (2.0 * sqrt_t) - rate * strike * math.exp(
            -rate * time_to_expiry
        ) * normal_cdf(d2)
    else:
        price = strike * math.exp(-rate * time_to_expiry) * normal_cdf(-d2) - spot * normal_cdf(-d1)
        delta = normal_cdf(d1) - 1.0
        theta = -(spot * normal_pdf(d1) * sigma) / (2.0 * sqrt_t) + rate * strike * math.exp(
            -rate * time_to_expiry
        ) * normal_cdf(-d2)

    gamma = normal_pdf(d1) / (spot * sigma * sqrt_t)
    vega = spot * normal_pdf(d1) * sqrt_t
    return {
        "price": price,
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega": vega,
    }


def solve_implied_volatility(
    observed_price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    option_type: str,
    low: float = 0.001,
    high: float = 5.0,
    tolerance: float = 1e-5,
    max_iterations: int = 100,
) -> float:
    if observed_price <= 0.0 or time_to_expiry <= 0.0:
        return float("nan")

    left = low
    right = high
    midpoint = (left + right) / 2.0
    for _ in range(max_iterations):
        midpoint = (left + right) / 2.0
        model_price = black_scholes_price_and_greeks(
            spot=spot,
            strike=strike,
            time_to_expiry=time_to_expiry,
            rate=rate,
            volatility=midpoint,
            option_type=option_type,
        )["price"]
        if abs(model_price - observed_price) < tolerance:
            return midpoint
        if model_price < observed_price:
            left = midpoint
        else:
            right = midpoint
    return midpoint
