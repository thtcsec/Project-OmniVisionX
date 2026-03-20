# OmniVision Simulator - Mock Camera RTSP Server
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY omni-simulator/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# Create videos directory for dropping video files
RUN mkdir -p /videos

COPY omni-simulator/ /app/

# HTTP API port + RTSP port (handled by ffmpeg)
EXPOSE 8554 8555

# Run both uvicorn (API) and RTSP server via ffmpeg
# RTSP URLs: rtsp://localhost:8554/<camera-id>
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8554"]
