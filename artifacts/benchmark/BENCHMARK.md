# Benchmark Report

- Benchmark run date: `2026-07-08T14:15:13.002077Z`
- Test set size: `12` paths
- Random seed: `42`

## Regime: gbm

| Model | Mean P&L | Std Dev | VaR 5% | CVaR 5% | Total Tx Cost | Downside Dev |
| --- | --- | --- | --- | --- | --- | --- |
| Black-Scholes delta (no tx cost) | -171.570465 | 10.072200 | -186.210281 | 186.210281 | 0.000000 | 171.865860 |
| Black-Scholes delta (with tx cost) | -378.220490 | 63.428982 | -484.977844 | 484.977844 | 2686.449707 | 383.502197 |
| deep_hedger_v1 | -130.814651 | 117.228340 | -453.047638 | 453.047638 | 0.000000 | 175.655792 |
| deep_hedger_v2 | -189.284943 | 172.071655 | -674.559998 | 674.559998 | 418.348694 | 255.807434 |
| deep_hedger_v3 | -157.205673 | 91.229874 | -387.780334 | 387.780334 | 477.885132 | 181.759506 |

### Improvement vs Black-Scholes baseline

| Model | Mean P&L | Std Dev | VaR 5% | CVaR 5% | Total Tx Cost | Downside Dev |
| --- | --- | --- | --- | --- | --- | --- |
| Metric Direction | higher better | lower better | higher better | lower better | lower better | lower better |
| deep_hedger_v1 | 23.75% | 1063.88% | -143.30% | 143.30% | n/a | 2.21% |
| deep_hedger_v2 | -10.32% | 1608.38% | -262.26% | 262.26% | n/a | 48.84% |
| deep_hedger_v3 | 8.37% | 805.76% | -108.25% | 108.25% | n/a | 5.76% |

## Regime: jump_diffusion

| Model | Mean P&L | Std Dev | VaR 5% | CVaR 5% | Total Tx Cost | Downside Dev |
| --- | --- | --- | --- | --- | --- | --- |
| Black-Scholes delta (no tx cost) | -194.030197 | 73.864922 | -449.042480 | 449.042480 | 0.000000 | 207.614426 |
| Black-Scholes delta (with tx cost) | -420.215973 | 67.040779 | -610.456909 | 610.456909 | 2940.415039 | 425.530182 |
| deep_hedger_v1 | -121.707047 | 94.722054 | -369.193909 | 369.193909 | 0.000000 | 154.223450 |
| deep_hedger_v2 | -180.234558 | 143.878983 | -589.818604 | 589.818604 | 443.950684 | 230.620163 |
| deep_hedger_v3 | -145.721710 | 77.019127 | -333.291077 | 333.291077 | 475.332336 | 164.823456 |

### Improvement vs Black-Scholes baseline

| Model | Mean P&L | Std Dev | VaR 5% | CVaR 5% | Total Tx Cost | Downside Dev |
| --- | --- | --- | --- | --- | --- | --- |
| Metric Direction | higher better | lower better | higher better | lower better | lower better | lower better |
| deep_hedger_v1 | 37.27% | 28.24% | 17.78% | -17.78% | n/a | -25.72% |
| deep_hedger_v2 | 7.11% | 94.79% | -31.35% | 31.35% | n/a | 11.08% |
| deep_hedger_v3 | 24.90% | 4.27% | 25.78% | -25.78% | n/a | -20.61% |

## Model Changes

### deep_hedger_v1
- Matches the notebook feature set: spot, Black-Scholes delta, previous hedge.
- Uses a single hidden layer with tanh activations.
- Optimizes terminal CVaR without transaction costs.
- Checkpoint status: available
- Features: `spot, bs_delta, prev_delta`
- Hidden dims: `[32]`
- Transaction cost rate: `0.0`
- Default regime: `gbm`

### deep_hedger_v2
- Adds normalized state features instead of raw spot only.
- Adds time-to-expiry and implied-volatility context to the hedge decision.
- Uses a two-layer MLP and includes transaction costs in training and evaluation.
- Checkpoint status: available
- Features: `log_moneyness, time_to_expiry, bs_delta, implied_volatility, prev_delta`
- Hidden dims: `[64, 64]`
- Transaction cost rate: `0.001`
- Default regime: `gbm`

### deep_hedger_v3
- Adds path-state features such as realized volatility, running P&L, and normalized step index.
- Uses a deeper three-layer MLP for a richer hedge policy without introducing recurrent layers.
- Intended to be trained on jump-diffusion paths while keeping the same benchmark interface.
- Checkpoint status: available
- Features: `log_moneyness, time_to_expiry, bs_delta, bs_gamma, bs_theta, bs_vega, implied_volatility, realized_volatility, step_fraction, running_pnl, prev_delta`
- Hidden dims: `[128, 128, 64]`
- Transaction cost rate: `0.001`
- Default regime: `jump_diffusion`
