#!/bin/bash
# omni-object Docker entrypoint
# Ensures YOLO model and cache dirs are available before starting the app

set -e

# Ensure weights dir exists (volume mount)
mkdir -p /app/weights 2>/dev/null || true

# Ensure ultralytics cache dir exists
mkdir -p /cache/ultralytics 2>/dev/null || true

# ── Copy pre-cached YOLO model if volume is empty ──
# The Dockerfile pre-downloads yolo11m.pt to /opt/yolo-cache/.
# If the volume mount (/app/weights) doesn't have it, copy from build cache.
YOLO_PATH="/app/weights/yolo11m.pt"
PRECACHE_PATH="/opt/yolo-cache/yolo11m.pt"
if [ ! -f "$YOLO_PATH" ]; then
    if [ -f "$PRECACHE_PATH" ]; then
        echo "📦 Copying pre-cached YOLO yolo11m.pt to volume..."
        if cp "$PRECACHE_PATH" "$YOLO_PATH" 2>&1; then
            echo "✅ YOLO model ready (from build cache)"
        else
            echo "❌ Failed to copy YOLO model — check volume permissions"
            echo "   Try: chmod 777 ./data/weights on the host"
        fi
        if [ ! -f "$YOLO_PATH" ]; then
            echo "⚠️ YOLO model NOT in volume after copy attempt — will download at runtime"
        fi
    else
        echo "⚠️ No pre-cached YOLO model — will download at runtime"
    fi
else
    echo "✅ YOLO model already in volume ($(du -h "$YOLO_PATH" | cut -f1))"
fi

# Execute the main command
exec "$@"
