#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-artifacts/checkpoints}"
BENCHMARK_DIR="${BENCHMARK_DIR:-artifacts/benchmark}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
JUMP_INTENSITY="${JUMP_INTENSITY:-120}"
JUMP_MEAN="${JUMP_MEAN:--0.03}"
JUMP_STD="${JUMP_STD:-0.12}"

mkdir -p "$CHECKPOINT_DIR" "$BENCHMARK_DIR"

echo "Running clean retrain + benchmark with run tag: $RUN_TAG"

CHECKPOINT_DIR="$CHECKPOINT_DIR" \
BENCHMARK_DIR="$BENCHMARK_DIR" \
bash scripts/clean_checkpoints.sh

RUN_TAG="$RUN_TAG" \
JUMP_INTENSITY="$JUMP_INTENSITY" \
JUMP_MEAN="$JUMP_MEAN" \
JUMP_STD="$JUMP_STD" \
CHECKPOINT_DIR="$CHECKPOINT_DIR" \
bash scripts/train_all_models.sh

RUN_TAG="$RUN_TAG" \
JUMP_INTENSITY="$JUMP_INTENSITY" \
JUMP_MEAN="$JUMP_MEAN" \
JUMP_STD="$JUMP_STD" \
CHECKPOINT_DIR="$CHECKPOINT_DIR" \
BENCHMARK_DIR="$BENCHMARK_DIR" \
bash scripts/run_benchmark.sh

echo "Finished retraining and benchmarking for run tag: $RUN_TAG"
