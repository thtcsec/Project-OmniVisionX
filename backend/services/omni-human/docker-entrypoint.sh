#!/bin/bash
# omni-human Docker entrypoint
# Fixes volume permissions and starts the application

set -e

# Disable albumentations update check
export NO_ALBUMENTATIONS_UPDATE=1

echo "🔧 Checking and fixing volume permissions..."

# 1. Ensure directories exist on the mounted volumes
mkdir -p /cache/insightface/models 2>/dev/null || true
mkdir -p /cache/matplotlib 2>/dev/null || true
mkdir -p /data/weights/human 2>/dev/null || true
mkdir -p /data/weights/face 2>/dev/null || true
mkdir -p /app/thumbnails 2>/dev/null || true

# 2. Fix ownership (since volumes are mounted from host, they might be root)
chown -R appuser:appuser /cache 2>/dev/null || true
chown -R appuser:appuser /data/weights 2>/dev/null || true
chown -R appuser:appuser /app/thumbnails 2>/dev/null || true

# 3. Setup Matplotlib cache explicitly to writable location
export MPLCONFIGDIR=/cache/matplotlib

# 4. Copy pre-cached InsightFace model if volume is empty
INSIGHT_CACHE="${INSIGHTFACE_HOME:-/cache/insightface}"
MODEL_DIR="$INSIGHT_CACHE/models/buffalo_l"
PRECACHE_DIR="/opt/insightface-cache/models/buffalo_l"

if [ ! -d "$MODEL_DIR" ] || [ -z "$(ls -A "$MODEL_DIR" 2>/dev/null)" ]; then
    if [ -d "$PRECACHE_DIR" ] && [ -n "$(ls -A "$PRECACHE_DIR" 2>/dev/null)" ]; then
        echo "📦 Copying pre-cached InsightFace buffalo_l model to volume..."
        cp -r "$PRECACHE_DIR" "$INSIGHT_CACHE/models/" 2>/dev/null || true
        chown -R appuser:appuser "$INSIGHT_CACHE/models/buffalo_l" 2>/dev/null || true
        echo "✅ InsightFace model ready (from build cache)"
    else
        echo "⚠️ No pre-cached model found — InsightFace will download at runtime"
    fi
else
    echo "✅ InsightFace model already in volume"
fi

# 5. Execute the main command (already running as appuser via Dockerfile USER)
echo "🚀 Starting omni-human..."
exec "$@"
