"""
Detection API Endpoints — omni-vehicle microservice version.
Adapted from root ai-engine/app/api/detection.py.

Uses fortress_lpr pipeline instead of ObjectDetector.
"""
import io
import asyncio
import base64
import binascii
import logging
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form
from pydantic import BaseModel
from PIL import Image, ImageOps, UnidentifiedImageError
import numpy as np
import cv2

from app.services.pipeline.orchestration.fortress_lpr import get_fortress_lpr

router = APIRouter()
logger = logging.getLogger("omni-vehicle.detection_api")

MAX_IMAGE_BYTES = 20 * 1024 * 1024


def _decode_image(data: bytes) -> tuple[Image.Image, np.ndarray]:
    image = Image.open(io.BytesIO(data))
    image = ImageOps.exif_transpose(image).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return image, img_bgr


class PlateDetectionResult(BaseModel):
    """Single plate detection result."""
    plate_text: str
    confidence: float
    bbox: list[int]  # [x1, y1, x2, y2] in pixels
    track_id: Optional[int] = None


class DetectionResponse(BaseModel):
    """Response from detection endpoint."""
    success: bool
    timestamp: str
    image_size: list[int]  # [width, height]
    plates: list[PlateDetectionResult]
    processing_time_ms: float


@router.post("/vehicle/detect", response_model=DetectionResponse)
async def detect_plates_from_file(
    file: UploadFile = File(...),
    is_night: bool = Form(False),
):
    """
    Detect and read license plates from an uploaded image.
    Uses the Fortress LPR pipeline.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Image payload is empty")
        if len(contents) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail="Image payload exceeds 20MB limit")
        image, img_bgr = _decode_image(contents)

        lpr = get_fortress_lpr()
        result = await asyncio.to_thread(lpr.process_frame, img_bgr)
        plates = []
        for plate in result.plates:
            plates.append(PlateDetectionResult(
                plate_text=plate.plate_text,
                confidence=round(float(plate.confidence), 3),
                bbox=list(plate.bbox) if plate.bbox else [0, 0, 0, 0],
                track_id=None,
            ))

        return DetectionResponse(
            success=True,
            timestamp=datetime.now().isoformat(),
            image_size=[image.width, image.height],
            plates=plates,
            processing_time_ms=round(result.total_time_ms, 2),
        )

    except HTTPException:
        raise
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image payload")
    except Exception:
        logger.exception("Detection failed")
        raise HTTPException(status_code=500, detail="Detection failed: internal error")


@router.post("/vehicle/detect/base64", response_model=DetectionResponse)
async def detect_plates_from_base64(
    image_base64: str,
    is_night: bool = False,
):
    """
    Detect plates from a base64-encoded image.
    Useful for internal service-to-service calls.
    """
    try:
        payload = image_base64.strip()
        if payload.startswith("data:"):
            payload = payload.split(",", 1)[-1]
        if not payload:
            raise HTTPException(status_code=400, detail="Image payload is empty")

        estimated_size = (len(payload) * 3) // 4
        if estimated_size > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail="Image payload exceeds 20MB limit")

        try:
            image_data = base64.b64decode(payload, validate=True)
        except binascii.Error as exc:
            raise HTTPException(status_code=400, detail="Invalid base64 image payload") from exc

        if not image_data:
            raise HTTPException(status_code=400, detail="Image payload is empty")
        if len(image_data) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail="Image payload exceeds 20MB limit")

        image, img_bgr = _decode_image(image_data)

        lpr = get_fortress_lpr()
        result = await asyncio.to_thread(lpr.process_frame, img_bgr)
        plates = []
        for plate in result.plates:
            plates.append(PlateDetectionResult(
                plate_text=plate.plate_text,
                confidence=round(float(plate.confidence), 3),
                bbox=list(plate.bbox) if plate.bbox else [0, 0, 0, 0],
                track_id=None,
            ))

        return DetectionResponse(
            success=True,
            timestamp=datetime.now().isoformat(),
            image_size=[image.width, image.height],
            plates=plates,
            processing_time_ms=round(result.total_time_ms, 2),
        )

    except HTTPException:
        raise
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image payload")
    except Exception:
        logger.exception("Detection failed")
        raise HTTPException(status_code=500, detail="Detection failed: internal error")
