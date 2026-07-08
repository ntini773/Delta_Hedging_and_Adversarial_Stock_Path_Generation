# Deep Hedging

Minimal PyTorch implementation of the workflow from `HFT_CAMP_CODE.ipynb`, split into small readable modules.

## File Layout

- `src/black_scholes.py`: Black-Scholes pricing, Greeks, and implied volatility solver
- `src/loss.py`: hedging P&L, transaction costs, CVaR, VaR, downside deviation
- `src/model.py`: deep hedger model versions and model specs
- `src/data.py`: CSV loading, symbol parsing, market context, GBM and jump-diffusion paths, feature builders
- `src/train.py`: train a selected model version and save checkpoint + metadata
- `src/benchmark.py`: load saved checkpoints only and write `BENCHMARK.md`

## Model Versions

- `v1`: notebook-style policy with inputs `spot`, `bs_delta`, `prev_delta`
- `v2`: adds `log_moneyness`, `time_to_expiry`, `implied_volatility`, and transaction costs
- `v3`: deeper MLP with richer state features and jump-diffusion as the default training regime

## Train Commands

Train `v1` on GBM:

```bash
python -m src.train --model-version v1 --regime gbm
```

Train `v2` on GBM:

```bash
python -m src.train --model-version v2 --regime gbm
```

Train `v3` on jump-diffusion:

```bash
python -m src.train --model-version v3 --regime jump_diffusion
```

Checkpoints are written to `artifacts/checkpoints/`.

## Benchmark Command

Benchmark every available checkpoint on both GBM and jump-diffusion test sets:

```bash
python -m src.benchmark
```

This writes:

- `artifacts/benchmark/BENCHMARK.md`
- `artifacts/benchmark/benchmark_pnl_arrays.npz`
- `artifacts/benchmark/benchmark_manifest.json`

## Bash Scripts

Train `v1`, `v2`, and `v3` in parallel on GPUs `0`, `1`, and `2`:

```bash
bash scripts/train_all_models.sh
```

Run the benchmark from saved checkpoints:

```bash
bash scripts/run_benchmark.sh
```

Both scripts accept overrides through environment variables such as `NUM_PATHS`, `EPOCHS`, `BATCH_SIZE`, `CHECKPOINT_DIR`, and `BENCHMARK_DIR`.

Default `NUM_PATHS` in both bash scripts is `10000`.

## Notes

- Default data file: `data/20260205_option_minute_prices_expiry.csv`
- The benchmark never retrains models. Missing checkpoints are reported explicitly.
- Transaction costs are part of P&L and the loss definition, not the raw CSV data.
