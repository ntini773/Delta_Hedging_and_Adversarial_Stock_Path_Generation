# Benchmark Report

- Benchmark run date: `2026-07-08T18:12:49.647458Z`
- Run tag filter: `20260708T174724Z`
- Dataset split: `train 70% / validation 10% / test 20%`
- Test set size: `2000` paths
- Random seed: `42`

## Regime: gbm

| Model | Mean P&L | Std Dev | VaR 5% | CVaR 5% | Total Tx Cost | Downside Dev |
| --- | --- | --- | --- | --- | --- | --- |
| Black-Scholes delta (no tx cost) | -171.638672 | 7.334928 | -183.736115 | 188.488556 | 0.000000 | 171.795334 |
| Black-Scholes delta (with tx cost) | -356.185577 | 62.367668 | -459.806732 | 480.486938 | 369093.781250 | 361.604614 |
| deep_hedger_v1 | -172.537842 | 125.143494 | -419.765747 | 498.829468 | 0.000000 | 213.143616 |
| deep_hedger_v3 | -219.756454 | 120.897491 | -430.775177 | 477.510559 | 93142.226562 | 250.816879 |

### Improvement vs Black-Scholes baseline

| Model | Mean P&L | Std Dev | VaR 5% | CVaR 5% | Total Tx Cost | Downside Dev |
| --- | --- | --- | --- | --- | --- | --- |
| Metric Direction | higher better | lower better | higher better | lower better | lower better | lower better |
| deep_hedger_v1 | -0.52% | 1606.13% | -128.46% | 164.65% | n/a | 24.07% |
| deep_hedger_v3 | -28.03% | 1548.24% | -134.45% | 153.34% | n/a | 46.00% |

Skipped checkpoints due to incompatible training config:
- deep_hedger_v2: validation_ratio mismatch (checkpoint=None, expected=0.1)

## Regime: jump_diffusion

| Model | Mean P&L | Std Dev | VaR 5% | CVaR 5% | Total Tx Cost | Downside Dev |
| --- | --- | --- | --- | --- | --- | --- |
| Black-Scholes delta (no tx cost) | -185.528702 | 141.394394 | -186.714508 | 460.628448 | 0.000000 | 233.266525 |
| Black-Scholes delta (with tx cost) | -371.278412 | 148.324982 | -469.017944 | 704.404785 | 371499.406250 | 399.809906 |
| deep_hedger_v1 | -185.935181 | 200.111816 | -445.761169 | 755.583069 | 0.000000 | 273.160431 |
| deep_hedger_v3 | -231.522842 | 193.646210 | -451.141968 | 738.889221 | 90735.039062 | 301.830536 |

### Improvement vs Black-Scholes baseline

| Model | Mean P&L | Std Dev | VaR 5% | CVaR 5% | Total Tx Cost | Downside Dev |
| --- | --- | --- | --- | --- | --- | --- |
| Metric Direction | higher better | lower better | higher better | lower better | lower better | lower better |
| deep_hedger_v1 | -0.22% | 41.53% | -138.74% | 64.03% | n/a | 17.10% |
| deep_hedger_v3 | -24.79% | 36.95% | -141.62% | 60.41% | n/a | 29.39% |

Skipped checkpoints due to incompatible training config:
- deep_hedger_v2: validation_ratio mismatch (checkpoint=None, expected=0.1)

## Model Changes

### deep_hedger_v1
- Matches the notebook feature set: spot, Black-Scholes delta, previous hedge.
- Uses a single hidden layer with tanh activations.
- Optimizes terminal CVaR without transaction costs.
- Checkpoint status: available
- Checkpoint type: best validation CVaR checkpoint
- Features: `spot, bs_delta, prev_delta`
- Hidden dims: `[32]`
- Transaction cost rate: `0.0`
- Default regime: `gbm`
- Run tag: `20260708T174724Z`

### deep_hedger_v2
- Adds normalized state features instead of raw spot only.
- Adds time-to-expiry and implied-volatility context to the hedge decision.
- Uses a two-layer MLP and includes transaction costs in training and evaluation.
- Checkpoint status: skipped
- Skip reason: validation_ratio mismatch (checkpoint=None, expected=0.1)
- Checkpoint type: best validation CVaR checkpoint
- Features: `log_moneyness, time_to_expiry, bs_delta, implied_volatility, prev_delta`
- Hidden dims: `[64, 64]`
- Transaction cost rate: `0.001`
- Default regime: `gbm`
- Run tag: ``

### deep_hedger_v3
- Adds path-state features such as realized volatility, running P&L, and normalized step index.
- Uses a deeper three-layer MLP for a richer hedge policy without introducing recurrent layers.
- Intended to be trained on jump-diffusion paths while keeping the same benchmark interface.
- Checkpoint status: available
- Checkpoint type: best validation CVaR checkpoint
- Features: `log_moneyness, time_to_expiry, bs_delta, bs_gamma, bs_theta, bs_vega, implied_volatility, realized_volatility, step_fraction, running_pnl, prev_delta`
- Hidden dims: `[128, 128, 64]`
- Transaction cost rate: `0.001`
- Default regime: `jump_diffusion`
- Run tag: `20260708T174724Z`
