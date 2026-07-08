# Benchmark Report

- Benchmark run date: `2026-07-08T17:37:47.370059Z`
- Run tag filter: `none`
- Dataset split: `train 70% / validation 10% / test 20%`
- Test set size: `12` paths
- Random seed: `42`

## Regime: gbm

| Model | Mean P&L | Std Dev | VaR 5% | CVaR 5% | Total Tx Cost | Downside Dev |
| --- | --- | --- | --- | --- | --- | --- |
| Black-Scholes delta (no tx cost) | -170.682053 | 10.458511 | -187.501984 | 187.501984 | 0.000000 | 171.002182 |
| Black-Scholes delta (with tx cost) | -364.813812 | 52.765953 | -454.223419 | 454.223419 | 2329.580566 | 368.609985 |
| deep_hedger_v1 | -215.967331 | 442.255280 | -1250.004639 | 1250.004639 | 0.000000 | 483.579895 |

### Improvement vs Black-Scholes baseline

| Model | Mean P&L | Std Dev | VaR 5% | CVaR 5% | Total Tx Cost | Downside Dev |
| --- | --- | --- | --- | --- | --- | --- |
| Metric Direction | higher better | lower better | higher better | lower better | lower better | lower better |
| deep_hedger_v1 | -26.53% | 4128.66% | -566.66% | 566.66% | n/a | 182.79% |

Skipped checkpoints due to incompatible training config:
- deep_hedger_v2: num_paths mismatch (checkpoint=64, expected=60)
- deep_hedger_v3: num_paths mismatch (checkpoint=64, expected=60)

## Regime: jump_diffusion

| Model | Mean P&L | Std Dev | VaR 5% | CVaR 5% | Total Tx Cost | Downside Dev |
| --- | --- | --- | --- | --- | --- | --- |
| Black-Scholes delta (no tx cost) | -229.699265 | 186.451920 | -847.791382 | 847.791382 | 0.000000 | 295.848053 |
| Black-Scholes delta (with tx cost) | -408.729980 | 167.845139 | -924.528076 | 924.528076 | 2148.369141 | 441.850861 |
| deep_hedger_v1 | -134.145233 | 476.021423 | -1499.945557 | 1499.945557 | 0.000000 | 465.713379 |

### Improvement vs Black-Scholes baseline

| Model | Mean P&L | Std Dev | VaR 5% | CVaR 5% | Total Tx Cost | Downside Dev |
| --- | --- | --- | --- | --- | --- | --- |
| Metric Direction | higher better | lower better | higher better | lower better | lower better | lower better |
| deep_hedger_v1 | 41.60% | 155.31% | -76.92% | 76.92% | n/a | 57.42% |

Skipped checkpoints due to incompatible training config:
- deep_hedger_v2: num_paths mismatch (checkpoint=64, expected=60)
- deep_hedger_v3: num_paths mismatch (checkpoint=64, expected=60)

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
- Run tag: ``

### deep_hedger_v2
- Adds normalized state features instead of raw spot only.
- Adds time-to-expiry and implied-volatility context to the hedge decision.
- Uses a two-layer MLP and includes transaction costs in training and evaluation.
- Checkpoint status: skipped
- Skip reason: num_paths mismatch (checkpoint=64, expected=60)
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
- Checkpoint status: skipped
- Skip reason: num_paths mismatch (checkpoint=64, expected=60)
- Checkpoint type: best validation CVaR checkpoint
- Features: `log_moneyness, time_to_expiry, bs_delta, bs_gamma, bs_theta, bs_vega, implied_volatility, realized_volatility, step_fraction, running_pnl, prev_delta`
- Hidden dims: `[128, 128, 64]`
- Transaction cost rate: `0.001`
- Default regime: `jump_diffusion`
- Run tag: ``
