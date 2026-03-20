"""
P0.5: Image Source Detector
===========================
Detect if image came from SDK (pre-processed) or raw camera stream.

Problem:
- Dahua SDK images already have HDR/sharpen applied
- Applying CLAHE + gamma + sharpen again = "artificial" looking images
- OCR models perform worse on over-enhanced images

Solution:
- Detect image source characteristics
- Skip aggressive enhancement for SDK images
- Apply full pipeline only for raw camera frames
"""

import numpy as np
import cv2
import logging
from typing import Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ImageSource(Enum):
    """Detected image source type"""
    SDK_PROCESSED = "sdk"       # Already enhanced by camera SDK
    RAW_CAMERA = "raw"          # Raw RTSP/snapshot frame
    UNKNOWN = "unknown"


@dataclass
class ImageAnalysis:
    """Image analysis results"""
    source: ImageSource
    sharpness: float           # Higher = sharper (likely SDK processed)
    contrast: float            # Higher = more contrast
    saturation: float          # Average saturation
    noise_level: float         # Estimated noise level
    has_jpeg_artifacts: bool   # JPEG compression artifacts
    confidence: float          # Confidence in source detection


def estimate_sharpness(img: np.ndarray) -> float:
    """
    Estimate image sharpness using Laplacian variance.
    Higher values = sharper image.
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return laplacian.var()


def estimate_contrast(img: np.ndarray) -> float:
    """
    Estimate image contrast using standard deviation of luminance.
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    return gray.std()


def estimate_noise(img: np.ndarray) -> float:
    """
    Estimate noise level using high-frequency component analysis.
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # High-pass filter to extract noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = cv2.absdiff(gray, blur)
    
    return noise.std()


def detect_jpeg_artifacts(img: np.ndarray) -> bool:
    """
    Detect JPEG compression artifacts (blocking).
    SDK images typically have less blocking than re-encoded frames.
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    h, w = gray.shape
    
    # Check for 8x8 block boundaries (JPEG block size)
    block_size = 8
    
    if h < block_size * 3 or w < block_size * 3:
        return False
    
    # Calculate differences at block boundaries vs inside blocks
    boundary_diffs = []
    inside_diffs = []
    
    for y in range(block_size, h - block_size, block_size):
        for x in range(block_size, w - block_size, block_size):
            # Boundary difference (between blocks)
            boundary_diffs.append(abs(int(gray[y, x]) - int(gray[y-1, x])))
            # Inside difference
            inside_diffs.append(abs(int(gray[y+2, x]) - int(gray[y+1, x])))
    
    if not boundary_diffs or not inside_diffs:
        return False
    
    avg_boundary = np.mean(boundary_diffs)
    avg_inside = np.mean(inside_diffs)
    
    # If boundary differences are notably higher, likely JPEG artifacts
    return avg_boundary > avg_inside * 1.5


def analyze_image(img: np.ndarray) -> ImageAnalysis:
    """
    Analyze image to determine source and characteristics.
    """
    if img is None or img.size == 0:
        return ImageAnalysis(
            source=ImageSource.UNKNOWN,
            sharpness=0, contrast=0, saturation=0,
            noise_level=0, has_jpeg_artifacts=False, confidence=0
        )
    
    # Calculate metrics
    sharpness = estimate_sharpness(img)
    contrast = estimate_contrast(img)
    noise = estimate_noise(img)
    has_artifacts = detect_jpeg_artifacts(img)
    
    # Calculate saturation
    if len(img.shape) == 3:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1].mean()
    else:
        saturation = 0
    
    # Determine source based on characteristics
    # SDK images typically have:
    # - Higher sharpness (already enhanced)
    # - Higher contrast
    # - Less noise (denoised)
    # - Fewer JPEG artifacts (better quality)
    
    sdk_score = 0.0
    
    # High sharpness suggests SDK processing
    if sharpness > 500:
        sdk_score += 0.3
    elif sharpness > 200:
        sdk_score += 0.15
    
    # High contrast suggests SDK processing
    if contrast > 60:
        sdk_score += 0.2
    elif contrast > 45:
        sdk_score += 0.1
    
    # Low noise suggests SDK processing
    if noise < 5:
        sdk_score += 0.25
    elif noise < 10:
        sdk_score += 0.1
    
    # No JPEG artifacts suggests SDK direct output
    if not has_artifacts:
        sdk_score += 0.25
    
    # Determine source
    if sdk_score >= 0.6:
        source = ImageSource.SDK_PROCESSED
        confidence = min(1.0, sdk_score)
    elif sdk_score <= 0.3:
        source = ImageSource.RAW_CAMERA
        confidence = min(1.0, 1.0 - sdk_score)
    else:
        source = ImageSource.UNKNOWN
        confidence = 0.5
    
    return ImageAnalysis(
        source=source,
        sharpness=sharpness,
        contrast=contrast,
        saturation=saturation,
        noise_level=noise,
        has_jpeg_artifacts=has_artifacts,
        confidence=confidence
    )


def should_skip_enhancement(img: np.ndarray, source_hint: str = None) -> bool:
    """
    P0.5 FIX: Determine if enhancement should be skipped.
    
    Args:
        img: Image to analyze
        source_hint: Optional hint ("dahua_sdk", "rtsp", etc.)
    
    Returns:
        True if enhancement should be skipped
    """
    # If source is explicitly SDK, skip enhancement
    if source_hint and "sdk" in source_hint.lower():
        logger.debug("Skipping enhancement: source hint indicates SDK image")
        return True
    
    # Analyze image
    analysis = analyze_image(img)
    
    if analysis.source == ImageSource.SDK_PROCESSED:
        logger.debug(f"Skipping enhancement: detected SDK image (conf={analysis.confidence:.2f})")
        return True
    
    return False


def get_enhancement_level(img: np.ndarray, source_hint: str = None) -> str:
    """
    Get recommended enhancement level for image.
    
    Returns:
        "none" - Skip all enhancement
        "light" - Only basic normalization
        "full" - Full enhancement pipeline
    """
    if should_skip_enhancement(img, source_hint):
        return "none"
    
    analysis = analyze_image(img)
    
    # If already decent quality, only light enhancement
    if analysis.sharpness > 100 and analysis.contrast > 40:
        return "light"
    
    return "full"
