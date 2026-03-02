#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${1:-doctoral_v1}"
OUT_DIR="${2:-artifacts}"
MODE="${3:---quick}"

PYTHONPATH=src python3 -m cross_asset_research.pipeline "$MODE" --run-name "$RUN_NAME" --output-dir "$OUT_DIR"
