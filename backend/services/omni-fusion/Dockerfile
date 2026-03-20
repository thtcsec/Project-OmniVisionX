# syntax=docker/dockerfile:1.4
# omni-fusion: spatial–temporal fusion (Redis consumer)
# Build context: ./backend (Project OmniVision)

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY services/omni-fusion/requirements.txt /app/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && pip install -r /app/requirements.txt

COPY services/omni-fusion/app /app/app

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8003/fusion/health || exit 1

EXPOSE 8003

RUN addgroup --system appuser && adduser --system --group appuser \
    && chown -R appuser:appuser /app

USER appuser

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8003"]
