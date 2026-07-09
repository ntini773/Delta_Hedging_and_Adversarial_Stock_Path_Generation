#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-artifacts/checkpoints}"
MODEL_VERSION="${MODEL_VERSION:-v3}"
CSV_PATH="${CSV_PATH:-data/20260205_option_minute_prices_expiry.csv}"
EXPIRY_TIME="${EXPIRY_TIME:-2026-02-05 15:30:00}"
REGIME="${REGIME:-}"
NUM_PATHS="${NUM_PATHS:-10000}"
VOLATILITY="${VOLATILITY:-0.6}"
RATE="${RATE:-0.05}"
SEED="${SEED:-42}"
TEST_RATIO="${TEST_RATIO:-0.2}"
VALIDATION_RATIO="${VALIDATION_RATIO:-0.1}"
PATH_INDEX="${PATH_INDEX:-0}"
DELAY="${DELAY:-0.10}"
DEVICE="${DEVICE:-cpu}"

python -m src.infer_tui \
  --checkpoints-dir "$CHECKPOINT_DIR" \
  --model-version "$MODEL_VERSION" \
  --csv-path "$CSV_PATH" \
  --expiry-time "$EXPIRY_TIME" \
  --regime "$REGIME" \
  --num-paths "$NUM_PATHS" \
  --volatility "$VOLATILITY" \
  --rate "$RATE" \
  --seed "$SEED" \
  --test-ratio "$TEST_RATIO" \
  --validation-ratio "$VALIDATION_RATIO" \
  --path-index "$PATH_INDEX" \
  --delay "$DELAY" \
  --device "$DEVICE"
