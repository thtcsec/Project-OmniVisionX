#!/bin/sh
set -e

WEIGHTS_PATH="${WEIGHTS_PATH:-/app/weights}"
PLATE_MODEL_PATH="${WEIGHTS_PATH}/LP_detector.pt"
PLATE_MODEL_URL="${LP_DETECTOR_URL:-https://huggingface.co/keremberke/yolov5n-license-plate/resolve/main/best.pt}"

mkdir -p "$WEIGHTS_PATH"
if [ ! -f "$PLATE_MODEL_PATH" ]; then
  if command -v curl >/dev/null 2>&1; then
    curl -L --retry 3 --retry-delay 2 -o "$PLATE_MODEL_PATH" "$PLATE_MODEL_URL" || true
  fi
fi

exec "$@"
