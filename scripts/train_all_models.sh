#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-artifacts/checkpoints}"
CSV_PATH="${CSV_PATH:-data/20260205_option_minute_prices_expiry.csv}"
EXPIRY_TIME="${EXPIRY_TIME:-2026-02-05 15:30:00}"
NUM_PATHS="${NUM_PATHS:-10000}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
SEED="${SEED:-42}"
VALIDATION_RATIO="${VALIDATION_RATIO:-0.1}"
TEST_RATIO="${TEST_RATIO:-0.2}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"

mkdir -p "$CHECKPOINT_DIR"
printf '%s\n' "$RUN_TAG" > "$CHECKPOINT_DIR/latest_run_tag.txt"

echo "Training v1 on GPU 0 (run tag: $RUN_TAG)"
CUDA_VISIBLE_DEVICES=0 python -m src.train \
  --model-version v1 \
  --regime gbm \
  --csv-path "$CSV_PATH" \
  --expiry-time "$EXPIRY_TIME" \
  --num-paths "$NUM_PATHS" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --learning-rate "$LEARNING_RATE" \
  --seed "$SEED" \
  --test-ratio "$TEST_RATIO" \
  --validation-ratio "$VALIDATION_RATIO" \
  --run-tag "$RUN_TAG" \
  --device cuda \
  --output-dir "$CHECKPOINT_DIR" &
PID_V1=$!

echo "Training v2 on GPU 1 (run tag: $RUN_TAG)"
CUDA_VISIBLE_DEVICES=1 python -m src.train \
  --model-version v2 \
  --regime gbm \
  --csv-path "$CSV_PATH" \
  --expiry-time "$EXPIRY_TIME" \
  --num-paths "$NUM_PATHS" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --learning-rate "$LEARNING_RATE" \
  --seed "$SEED" \
  --test-ratio "$TEST_RATIO" \
  --validation-ratio "$VALIDATION_RATIO" \
  --run-tag "$RUN_TAG" \
  --device cuda \
  --output-dir "$CHECKPOINT_DIR" &
PID_V2=$!

echo "Training v3 on GPU 2 (run tag: $RUN_TAG)"
CUDA_VISIBLE_DEVICES=2 python -m src.train \
  --model-version v3 \
  --regime jump_diffusion \
  --csv-path "$CSV_PATH" \
  --expiry-time "$EXPIRY_TIME" \
  --num-paths "$NUM_PATHS" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --learning-rate "$LEARNING_RATE" \
  --seed "$SEED" \
  --test-ratio "$TEST_RATIO" \
  --validation-ratio "$VALIDATION_RATIO" \
  --run-tag "$RUN_TAG" \
  --device cuda \
  --output-dir "$CHECKPOINT_DIR" &
PID_V3=$!

wait "$PID_V1"
wait "$PID_V2"
wait "$PID_V3"

echo "All training jobs finished."
