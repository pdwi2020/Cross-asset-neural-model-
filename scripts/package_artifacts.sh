#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${1:-doctoral_v1}"
OUT_DIR="${2:-artifacts}"

ROOT="${OUT_DIR%/}/${RUN_NAME}"
if [ ! -d "$ROOT" ]; then
  echo "Run directory not found: $ROOT" >&2
  exit 1
fi

ZIP_PATH="${OUT_DIR%/}/${RUN_NAME}_artifacts.zip"
rm -f "$ZIP_PATH"
(
  cd "${OUT_DIR%/}"
  zip -rq "$(basename "$ZIP_PATH")" "$RUN_NAME"
)

echo "$ZIP_PATH"
