"""
OmniVision LPR Service - BOILERPLATE
====================================
Responsibilities:
1. OmniPlateEngine: Vietnamese license plate detection + OCR
2. Redis Consumer: Read from omni:detections stream
3. Redis Publisher: Push plate events to omni:vehicles stream (when wired)
"""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("omni-lpr")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 OmniLPR Starting (BOILERPLATE)...")

    # TODO: Initialize components
    # - OmniPlateEngine (plate detector + OCR)
    # - Redis consumer (read from omni:detections)
    # - Redis publisher (push to omni:vehicles)

    logger.info("✅ OmniLPR Ready!")

    yield

    logger.info("👋 OmniLPR stopped.")


app = FastAPI(
    title="OmniVision LPR Service",
    description="Vietnamese License Plate Recognition",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/lpr/health")
async def health():
    return {"status": "boilerplate", "service": "omni-lpr", "version": "1.0.0"}


@app.get("/lpr/stats")
async def stats():
    return {"plates_detected": 0, "plates_ocr": 0}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
