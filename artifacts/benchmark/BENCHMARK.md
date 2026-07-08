# Benchmark Report

- Benchmark run date: `2026-07-08T17:09:26.888599Z`
- Test set size: `2000` paths
- Random seed: `42`

## Regime: gbm

| Model | Mean P&L | Std Dev | VaR 5% | CVaR 5% | Total Tx Cost | Downside Dev |
| --- | --- | --- | --- | --- | --- | --- |
| Black-Scholes delta (no tx cost) | -171.638672 | 7.334928 | -183.736115 | 188.488556 | 0.000000 | 171.795334 |
| Black-Scholes delta (with tx cost) | -356.185577 | 62.367668 | -459.806732 | 480.486938 | 369093.781250 | 361.604614 |
| deep_hedger_v1 | -171.998672 | 124.875610 | -414.773651 | 500.483002 | 0.000000 | 212.549896 |
| deep_hedger_v2 | -254.397781 | 40.745171 | -314.750122 | 332.162201 | 164136.609375 | 257.640045 |
| deep_hedger_v3 | -214.576782 | 132.118073 | -459.294434 | 539.447144 | 80176.109375 | 251.988861 |

### Improvement vs Black-Scholes baseline

| Model | Mean P&L | Std Dev | VaR 5% | CVaR 5% | Total Tx Cost | Downside Dev |
| --- | --- | --- | --- | --- | --- | --- |
| Metric Direction | higher better | lower better | higher better | lower better | lower better | lower better |
| deep_hedger_v1 | -0.21% | 1602.48% | -125.74% | 165.52% | n/a | 23.72% |
| deep_hedger_v2 | -48.22% | 455.50% | -71.31% | 76.22% | n/a | 49.97% |
| deep_hedger_v3 | -25.02% | 1701.22% | -149.98% | 186.20% | n/a | 46.68% |

## Regime: jump_diffusion

| Model | Mean P&L | Std Dev | VaR 5% | CVaR 5% | Total Tx Cost | Downside Dev |
| --- | --- | --- | --- | --- | --- | --- |
| Black-Scholes delta (no tx cost) | -185.528702 | 141.394394 | -186.714508 | 460.628448 | 0.000000 | 233.266525 |
| Black-Scholes delta (with tx cost) | -371.278412 | 148.324982 | -469.017944 | 704.404785 | 371499.406250 | 399.809906 |
| deep_hedger_v1 | -185.612167 | 203.056000 | -452.283630 | 762.919617 | 0.000000 | 275.106567 |
| deep_hedger_v2 | -267.810883 | 155.271347 | -322.567688 | 616.540405 | 164466.500000 | 309.567200 |
| deep_hedger_v3 | -225.932922 | 196.791885 | -503.296997 | 778.511414 | 78188.523438 | 299.620972 |

### Improvement vs Black-Scholes baseline

| Model | Mean P&L | Std Dev | VaR 5% | CVaR 5% | Total Tx Cost | Downside Dev |
| --- | --- | --- | --- | --- | --- | --- |
| Metric Direction | higher better | lower better | higher better | lower better | lower better | lower better |
| deep_hedger_v1 | -0.04% | 43.61% | -142.23% | 65.63% | n/a | 17.94% |
| deep_hedger_v2 | -44.35% | 9.81% | -72.76% | 33.85% | n/a | 32.71% |
| deep_hedger_v3 | -21.78% | 39.18% | -169.55% | 69.01% | n/a | 28.45% |

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
