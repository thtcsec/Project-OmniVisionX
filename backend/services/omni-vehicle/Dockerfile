# syntax=docker/dockerfile:1.4
# omni-vehicle: License plate recognition (LPR) service
# Build context: ./backend (Project OmniVision)

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV OMP_NUM_THREADS=1

# Install Python and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    ffmpeg \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip (cached)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip wheel && pip install "setuptools==69.5.1"

# Install PyTorch with CUDA (cached - large download)
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch torchvision --index-url ${TORCH_INDEX_URL}

# Pin numpy < 2.0 FIRST to avoid compatibility issues
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "numpy<2.0"

# Install dependencies from requirements.txt (single source of truth)
COPY services/omni-vehicle/requirements.txt /app/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /app/requirements.txt

WORKDIR /app

# Copy application code LAST (so code changes don't invalidate cache)
COPY services/omni-vehicle/app /app/app

# Training worker expects these paths (populate via volume or future commits)
RUN mkdir -p /app/scripts /app/training

# Create directories for runtime
RUN mkdir -p /app/weights /app/config /app/data /app/data/collect \
    /app/thumbnails /app/checkpoints /app/logs

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/vehicle/health || exit 1

EXPOSE 8001

# Set Python path
ENV PYTHONPATH="/opt/venv/lib/python3.10/site-packages:$PYTHONPATH"

# Create non-root user
RUN addgroup --system appuser && adduser --system --group appuser \
    && chown -R appuser:appuser /app \
    && mkdir -p /home/appuser/.cache /home/appuser/.paddleocr \
    && chown -R appuser:appuser /home/appuser

# Ensure permissions for writable directories
RUN chmod -R 755 /app
RUN chmod -R 777 /app/thumbnails /app/data /app/checkpoints /app/logs
RUN chmod -R 777 /home/appuser/.cache
RUN mkdir -p /home/appuser/.paddleocr/whl && chmod -R 777 /home/appuser/.paddleocr
RUN chmod 1777 /tmp

# Install Dahua SDK if mounted (runtime install via entrypoint)
COPY services/omni-vehicle/docker-entrypoint.sh /docker-entrypoint.sh
RUN sed -i 's/\r$//' /docker-entrypoint.sh && chmod +x /docker-entrypoint.sh

USER appuser

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
