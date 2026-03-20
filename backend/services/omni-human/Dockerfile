# syntax=docker/dockerfile:1.4
# omni-human — Build context: ./backend (Project OmniVision)

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install Python and dependencies
# Note: Adding Ubuntu Toolchain PPA for newer libstdc++6 (GLIBCXX_3.4.32 required by insightface)
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository -y ppa:ubuntu-toolchain-r/test \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libstdc++6 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip (cached)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA (cached - large download for ultralytics)
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch torchvision --index-url ${TORCH_INDEX_URL}

# Install dependencies from requirements.txt (single source of truth)
COPY services/omni-human/requirements.txt /app/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /app/requirements.txt

# Force numpy < 2.0 after all installs to avoid _ARRAY_API ABI errors
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "numpy<2.0"

WORKDIR /app

# ── Pre-download InsightFace buffalo_l model during build ──
# Saves ~281MB download at every container startup.
# Model is cached at /opt/insightface-cache/ (NOT in /cache which is a volume mount).
# The entrypoint copies it to the volume on first boot.
RUN mkdir -p /opt/insightface-cache/models && \
    python -c "\
    from insightface.app import FaceAnalysis; \
    app = FaceAnalysis(name='buffalo_l', root='/opt/insightface-cache', providers=['CPUExecutionProvider']); \
    app.prepare(ctx_id=-1, det_size=(640, 640)); \
    print('✅ InsightFace buffalo_l pre-cached')" \
    || echo "⚠️ InsightFace pre-download failed — will download at runtime"

# Copy application code LAST (so code changes don't invalidate cache)
COPY services/omni-human/app /app/app

# Create directories for InsightFace models
RUN mkdir -p /app/models /cache/insightface/models /app/thumbnails

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8002/human/health || exit 1

EXPOSE 8002

# Copy entrypoint
COPY services/omni-human/docker-entrypoint.sh /app/docker-entrypoint.sh

# Normalize line endings so Linux can execute the script
RUN sed -i 's/\r$//' /app/docker-entrypoint.sh

# Create non-root user with access to cache and thumbnail dirs
RUN addgroup --system appuser && adduser --system --group appuser \
    && chown -R appuser:appuser /app \
    && chown -R appuser:appuser /cache \
    && mkdir -p /home/appuser/.insightface \
    && chown -R appuser:appuser /home/appuser \
    && chmod +x /app/docker-entrypoint.sh

USER appuser

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]
