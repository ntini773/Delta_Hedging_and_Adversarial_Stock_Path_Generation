#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-artifacts/checkpoints}"
BENCHMARK_DIR="${BENCHMARK_DIR:-artifacts/benchmark}"
CSV_PATH="${CSV_PATH:-data/20260205_option_minute_prices_expiry.csv}"
EXPIRY_TIME="${EXPIRY_TIME:-2026-02-05 15:30:00}"
NUM_PATHS="${NUM_PATHS:-10000}"
SEED="${SEED:-42}"
VALIDATION_RATIO="${VALIDATION_RATIO:-0.1}"
TEST_RATIO="${TEST_RATIO:-0.2}"
RUN_TAG="${RUN_TAG:-}"

mkdir -p "$BENCHMARK_DIR"

if [[ -z "$RUN_TAG" && -f "$CHECKPOINT_DIR/latest_run_tag.txt" ]]; then
  RUN_TAG="$(cat "$CHECKPOINT_DIR/latest_run_tag.txt")"
fi

CUDA_VISIBLE_DEVICES="" python -m src.benchmark \
  --checkpoints-dir "$CHECKPOINT_DIR" \
  --output-dir "$BENCHMARK_DIR" \
  --csv-path "$CSV_PATH" \
  --expiry-time "$EXPIRY_TIME" \
  --num-paths "$NUM_PATHS" \
  --seed "$SEED" \
  --test-ratio "$TEST_RATIO" \
  --validation-ratio "$VALIDATION_RATIO" \
  --run-tag "$RUN_TAG" \
  --device cpu

echo "Benchmark report written to $BENCHMARK_DIR/BENCHMARK.md"
