# syntax=docker/dockerfile:1.4
# omni-object: RTSP flow + YOLO (YOLOv11 + ByteTrack) + Redis Streams
# Build context: repository ROOT (docker-compose context: .) — includes shared/

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV OMP_NUM_THREADS=1

# Install Python and runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Create venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch torchvision --index-url ${TORCH_INDEX_URL}

# Pin numpy
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "numpy<2.0"

# Install requirements
COPY backend/services/omni-object/requirements.txt /app/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /app/requirements.txt

WORKDIR /app

# ── Pre-download YOLO yolo11m.pt during build (~39MB) ──
RUN mkdir -p /opt/yolo-cache && \
    python -c "\
from ultralytics import YOLO; import shutil, os; \
m = YOLO('yolo11m.pt'); \
src = str(getattr(m, 'ckpt_path', '')); \
dst = '/opt/yolo-cache/yolo11m.pt'; \
if src and os.path.isfile(src): shutil.copy2(src, dst); print(f'✅ Copied {src} → {dst}'); \
else: print('⚠️ ckpt_path not found')" \
    || echo "⚠️ YOLO pre-download failed — will download at runtime"

# Copy application code and shared utilities (shared/ lives at repo root)
COPY backend/services/omni-object/app /app/app
COPY shared /app/shared

# Create necessary directories
RUN mkdir -p /app/weights /cache

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8555/rtsp/health || exit 1

EXPOSE 8555

ENV PYTHONPATH="/app:/opt/venv/lib/python3.11/site-packages:$PYTHONPATH"

# Copy entrypoint
COPY backend/services/omni-object/docker-entrypoint.sh /app/docker-entrypoint.sh

# Normalize line endings so Linux can execute the script
RUN sed -i 's/\r$//' /app/docker-entrypoint.sh

# Non-root user with access to weights and cache dirs
RUN addgroup --system appuser && adduser --system --group appuser \
    && chown -R appuser:appuser /app \
    && chown -R appuser:appuser /cache \
    && mkdir -p /home/appuser/.cache \
    && chown -R appuser:appuser /home/appuser \
    && chmod +x /app/docker-entrypoint.sh

RUN chmod -R 755 /app
RUN chmod 1777 /tmp

USER appuser

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8555"]
