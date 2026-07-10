# Benchmark Report

- Benchmark run date: `2026-07-10T08:44:45.199943Z`
- Run tag filter: `20260709T175659Z`
- Dataset split: `train 70% / validation 10% / test 20%`
- Test set size: `2000` paths
- Random seed: `42`
- Jump-diffusion intensity: `120.0` jumps/year when regime=`jump_diffusion`
- Jump mean / std in log space: `-0.03 / 0.12`

## Regime: gbm

| Model | Mean P&L | Std Dev | VaR 5% | Tail Loss 5% | Total Tx Cost | Downside Dev |
| --- | --- | --- | --- | --- | --- | --- |
| Black-Scholes delta (no tx cost) | -171.734329 | 7.490638 | -184.198608 | 189.053284 | 0.000000 | 171.897598 |
| Black-Scholes delta (with tx cost) | -357.134949 | 62.224152 | -460.166138 | 478.545044 | 370801.218750 | 362.515106 |
| deep_hedger_v1 | -197.393631 | 125.358955 | -439.624268 | 521.295654 | 49807.734375 | 233.835648 |
| deep_hedger_v2 | -254.402817 | 22.332090 | -291.086182 | 301.065552 | 165405.234375 | 255.381104 |
| deep_hedger_v3 | -212.625565 | 168.156097 | -562.101318 | 679.858276 | 83611.132812 | 271.083191 |
| deep_hedger_v4 | -295.623810 | 98.541603 | -470.040344 | 526.216064 | 246097.656250 | 311.614929 |

### Improvement vs Matched Black-Scholes baseline

| Model | Baseline | Mean P&L | Std Dev | VaR 5% | Tail Loss 5% | Total Tx Cost | Downside Dev |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Metric Direction | baseline-matched | higher better | lower better | higher better | lower better | lower better | lower better |
| deep_hedger_v1 | Black-Scholes delta (with tx cost) | 44.73% | -101.46% | 4.46% | -8.93% | 86.57% | 35.50% |
| deep_hedger_v2 | Black-Scholes delta (with tx cost) | 28.77% | 64.11% | 36.74% | 37.09% | 55.39% | 29.55% |
| deep_hedger_v3 | Black-Scholes delta (with tx cost) | 40.46% | -170.24% | -22.15% | -42.07% | 77.45% | 25.22% |
| deep_hedger_v4 | Black-Scholes delta (with tx cost) | 17.22% | -58.37% | -2.15% | -9.96% | 33.63% | 14.04% |

## Regime: jump_diffusion

| Model | Mean P&L | Std Dev | VaR 5% | Tail Loss 5% | Total Tx Cost | Downside Dev |
| --- | --- | --- | --- | --- | --- | --- |
| Black-Scholes delta (no tx cost) | -265.333160 | 468.709686 | -662.192871 | 1940.984253 | 0.000000 | 538.600464 |
| Black-Scholes delta (with tx cost) | -446.040436 | 460.707001 | -758.334778 | 2054.985840 | 361414.437500 | 641.251099 |
| deep_hedger_v1 | -295.060303 | 426.547913 | -994.149353 | 1899.713135 | 49798.878906 | 518.655701 |
| deep_hedger_v2 | -349.937592 | 454.919037 | -774.968262 | 2015.367310 | 164604.000000 | 573.940491 |
| deep_hedger_v3 | -303.729889 | 438.351746 | -799.172852 | 1715.035889 | 80133.546875 | 533.295532 |
| deep_hedger_v4 | -382.259521 | 400.011505 | -722.987854 | 1772.679565 | 237314.921875 | 553.291565 |

### Improvement vs Matched Black-Scholes baseline

| Model | Baseline | Mean P&L | Std Dev | VaR 5% | Tail Loss 5% | Total Tx Cost | Downside Dev |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Metric Direction | baseline-matched | higher better | lower better | higher better | lower better | lower better | lower better |
| deep_hedger_v1 | Black-Scholes delta (with tx cost) | 33.85% | 7.41% | -31.10% | 7.56% | 86.22% | 19.12% |
| deep_hedger_v2 | Black-Scholes delta (with tx cost) | 21.55% | 1.26% | -2.19% | 1.93% | 54.46% | 10.50% |
| deep_hedger_v3 | Black-Scholes delta (with tx cost) | 31.91% | 4.85% | -5.39% | 16.54% | 77.83% | 16.84% |
| deep_hedger_v4 | Black-Scholes delta (with tx cost) | 14.30% | 13.17% | 4.66% | 13.74% | 34.34% | 13.72% |

## Model Changes

### deep_hedger_v1
- Matches the notebook feature set: spot, Black-Scholes delta, previous hedge.
- Uses a single hidden layer with tanh activations.
- Optimizes terminal CVaR with the same transaction-cost model used by the benchmark.
- Checkpoint status: available
- Checkpoint type: best validation CVaR checkpoint
- Features: `spot, bs_delta, prev_delta`
- Hidden dims: `[32]`
- Transaction cost rate: `0.001`
- Default regime: `gbm`
- Run tag: `20260709T175659Z`

### deep_hedger_v2
- Adds normalized state features instead of raw spot only.
- Adds time-to-expiry and implied-volatility context to the hedge decision.
- Uses a two-layer MLP and includes transaction costs in training and evaluation.
- Checkpoint status: available
- Checkpoint type: best validation CVaR checkpoint
- Features: `log_moneyness, time_to_expiry, bs_delta, implied_volatility, prev_delta`
- Hidden dims: `[64, 64]`
- Transaction cost rate: `0.001`
- Default regime: `gbm`
- Run tag: `20260709T175659Z`

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
- Jump params: intensity=120.0, mean=-0.03, std=0.12
- Run tag: `20260709T175659Z`

### deep_hedger_v4
- Predicts a residual adjustment on top of Black-Scholes delta instead of an unconstrained absolute hedge.
- Uses normalized market-state features so the policy is less likely to collapse to a flat hedge.
- Adds alignment and turnover regularization to keep training stable on large synthetic batches.
- Checkpoint status: available
- Checkpoint type: best validation CVaR checkpoint
- Features: `log_moneyness, sqrt_time_to_expiry, bs_delta, delta_gap_to_prev, scaled_bs_gamma, scaled_bs_theta, scaled_bs_vega, implied_volatility, realized_volatility, instant_log_return, running_pnl_scaled, step_fraction, prev_delta`
- Hidden dims: `[256, 256, 128]`
- Transaction cost rate: `0.001`
- Default regime: `jump_diffusion`
- Jump params: intensity=120.0, mean=-0.03, std=0.12
- Run tag: `20260709T175659Z`
