# Deep Hedging

Minimal PyTorch implementation of the workflow from `HFT_CAMP_CODE.ipynb`, split into small readable modules.

## File Layout

- `src/black_scholes.py`: Black-Scholes pricing, Greeks, and implied volatility solver
- `src/loss.py`: hedging P&L, transaction costs, CVaR, VaR, downside deviation
- `src/model.py`: deep hedger model versions and model specs
- `src/data.py`: CSV loading, symbol parsing, market context, GBM and jump-diffusion paths, feature builders
- `src/train.py`: train a selected model version and save checkpoint + metadata
- `src/benchmark.py`: load saved checkpoints only and write `BENCHMARK.md`
- `src/infer_tui.py`: Rich terminal dashboard for single-path inference on the benchmark test split
- `docs/gbm_vs_jump.svg`: visual explainer for smooth GBM paths vs jump-diffusion stress paths
- `docs/benchmark_metrics_guide.svg`: visual explainer for benchmark metrics and fair comparisons

## Model Versions

- `v1`: notebook-style policy with inputs `spot`, `bs_delta`, `prev_delta`
- `v2`: adds `log_moneyness`, `time_to_expiry`, `implied_volatility`, and transaction costs
- `v3`: deeper MLP with richer state features and jump-diffusion as the default training regime
- `v4`: residual hedge around Black-Scholes delta with normalized features and regularized training

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

Train `v4` on jump-diffusion:

```bash
python -m src.train --model-version v4 --regime jump_diffusion
```

Checkpoints are written to `artifacts/checkpoints/`.
The canonical checkpoint name, such as `deep_hedger_v2.pt`, is the best checkpoint selected by validation CVaR.
The final-epoch copy is also saved as `*_last.pt` for inspection.

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

Train `v1`, `v2`, `v3`, and `v4` with a 3-GPU schedule:

```bash
bash scripts/train_all_models.sh
```

Run the benchmark from saved checkpoints:

```bash
bash scripts/run_benchmark.sh
```

Wipe old checkpoints and benchmark artifacts:

```bash
bash scripts/clean_checkpoints.sh
```

Run a full clean retrain + benchmark flow:

```bash
bash scripts/rerun_training.sh
```

Run the Rich TUI for one test-set path:

```bash
python -m src.infer_tui --model-version v3 --path-index 0
```

Or with the helper script:

```bash
bash scripts/run_inference_tui.sh
```

Both scripts accept overrides through environment variables such as `NUM_PATHS`, `EPOCHS`, `BATCH_SIZE`, `CHECKPOINT_DIR`, and `BENCHMARK_DIR`.
For Weights & Biases logging, set `WANDB_PROJECT`, optionally `WANDB_ENTITY`, and `WANDB_MODE=online` before running training.
Jump stress can be configured with `JUMP_INTENSITY`, `JUMP_MEAN`, and `JUMP_STD`. Current stronger defaults are `120`, `-0.03`, and `0.12`.

Default `NUM_PATHS` in both bash scripts is `10000`.
The training script writes a shared `RUN_TAG`, and the benchmark script reuses it so only checkpoints from the same coordinated run are compared by default.

## Notes

- Default data file: `data/20260205_option_minute_prices_expiry.csv`
- Default split: `70% train / 10% validation / 20% test`
- Visual guides: `docs/gbm_vs_jump.svg` and `docs/benchmark_metrics_guide.svg`
- The benchmark never retrains models. Missing checkpoints are reported explicitly.
- Benchmarking uses the canonical `deep_hedger_v*.pt` checkpoints, which are the best-validation checkpoints.
- Transaction costs are part of P&L and the loss definition, not the raw CSV data.
