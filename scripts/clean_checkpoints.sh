#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-artifacts/checkpoints}"
BENCHMARK_DIR="${BENCHMARK_DIR:-artifacts/benchmark}"

echo "Removing old checkpoints from $CHECKPOINT_DIR"
rm -f \
  "$CHECKPOINT_DIR"/deep_hedger_v1.pt \
  "$CHECKPOINT_DIR"/deep_hedger_v1.json \
  "$CHECKPOINT_DIR"/deep_hedger_v1_last.pt \
  "$CHECKPOINT_DIR"/deep_hedger_v1_last.json \
  "$CHECKPOINT_DIR"/deep_hedger_v2.pt \
  "$CHECKPOINT_DIR"/deep_hedger_v2.json \
  "$CHECKPOINT_DIR"/deep_hedger_v2_last.pt \
  "$CHECKPOINT_DIR"/deep_hedger_v2_last.json \
  "$CHECKPOINT_DIR"/deep_hedger_v3.pt \
  "$CHECKPOINT_DIR"/deep_hedger_v3.json \
  "$CHECKPOINT_DIR"/deep_hedger_v3_last.pt \
  "$CHECKPOINT_DIR"/deep_hedger_v3_last.json \
  "$CHECKPOINT_DIR"/deep_hedger_v4.pt \
  "$CHECKPOINT_DIR"/deep_hedger_v4.json \
  "$CHECKPOINT_DIR"/deep_hedger_v4_last.pt \
  "$CHECKPOINT_DIR"/deep_hedger_v4_last.json \
  "$CHECKPOINT_DIR"/latest_run_tag.txt

echo "Removing old benchmark outputs from $BENCHMARK_DIR"
rm -f \
  "$BENCHMARK_DIR"/BENCHMARK.md \
  "$BENCHMARK_DIR"/benchmark_pnl_arrays.npz \
  "$BENCHMARK_DIR"/benchmark_manifest.json

echo "Cleanup complete."
