"""
PHASE 1: Adaptive 2-Line Plate Splitter
========================================
Uses horizontal projection + valley detection instead of fixed 45/55 split.

Vietnamese 2-line plates:
- Line 1: Province code + Series (e.g., "29A")
- Line 2: Serial number (e.g., "12345")

The split position varies based on:
- Plate aspect ratio
- Font size differences between lines
- Distortion/perspective
"""

import numpy as np
import cv2
import logging
from typing import Tuple, Optional, List

logger = logging.getLogger(__name__)


def compute_horizontal_projection(img: np.ndarray) -> np.ndarray:
    """
    Compute horizontal projection (sum of pixels per row).
    Used to find text line boundaries.

    Args:
        img: Grayscale or binary image

    Returns:
        1D array of row sums (height,)
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Binarize with Otsu
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Sum pixels per row
    projection = np.sum(binary, axis=1)

    return projection


def find_split_valley(projection: np.ndarray,
                      min_valley_ratio: float = 0.3,
                      search_range: Tuple[float, float] = (0.3, 0.7)) -> Optional[int]:
    """
    Find the best split point (valley) in horizontal projection.

    Args:
        projection: Horizontal projection array
        min_valley_ratio: Minimum valley depth relative to peaks
        search_range: (min_ratio, max_ratio) of height to search

    Returns:
        Split row index, or None if no clear valley found
    """
    h = len(projection)

    if h < 20:
        return None

    # Search in middle region only
    start_row = int(h * search_range[0])
    end_row = int(h * search_range[1])

    if end_row <= start_row:
        return None

    search_region = projection[start_row:end_row]

    if len(search_region) == 0:
        return None

    # Find minimum in search region
    min_idx = np.argmin(search_region)
    min_val = search_region[min_idx]

    # Check if it's a significant valley
    # Compare to peaks before and after
    region_max = np.max(search_region)

    if region_max == 0:
        return None

    valley_depth = 1.0 - (min_val / region_max)

    if valley_depth >= min_valley_ratio:
        split_row = start_row + min_idx
        logger.debug("Valley found at row %s (depth=%.2f)", split_row, valley_depth)
        return split_row

    return None


def adaptive_split_2line(plate_img: np.ndarray,
                         fallback_ratio: float = 0.45) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    PHASE 1: Adaptively split a 2-line plate image.

    Uses horizontal projection to find the natural split point
    instead of fixed 45/55 ratio.

    Args:
        plate_img: Plate image (BGR or grayscale)
        fallback_ratio: Fallback split ratio if no valley found

    Returns:
        (line1_img, line2_img, split_row)
    """
    h, w = plate_img.shape[:2]

    if h < 30 or w < 50:
        # Too small, use fallback
        split_row = int(h * fallback_ratio)
        return plate_img[:split_row], plate_img[split_row:], split_row

    # Compute horizontal projection
    projection = compute_horizontal_projection(plate_img)

    # Smooth projection to reduce noise
    kernel_size = max(3, h // 15)
    if kernel_size % 2 == 0:
        kernel_size += 1
    # projection is reshaped to (-1, 1) column vector: blur along rows (axis 0)
    projection_smooth = cv2.GaussianBlur(
        projection.astype(np.float32).reshape(-1, 1),
        (kernel_size, 1),
        0
    ).flatten()

    # Find valley (split point)
    split_row = find_split_valley(projection_smooth)

    if split_row is None:
        # No clear valley, use fallback ratio
        split_row = int(h * fallback_ratio)
        logger.debug("No valley found, using fallback split at %s/%s", split_row, h)
    else:
        logger.debug("Adaptive split at row %s/%s (%.1f%%)", split_row, h, split_row / h * 100)

    # Extract lines
    line1 = plate_img[:split_row]
    line2 = plate_img[split_row:]

    return line1, line2, split_row


def is_2line_plate(plate_img: np.ndarray, aspect_threshold: float = 3.2) -> bool:
    """
    Detect if plate is likely a 2-line format.

    Vietnamese plates (2025+):
    - Wide/long plates: aspect ratio > ~3.2
    - Compact plates: aspect ratio <= ~3.2 (typically 2-line layouts)

    Args:
        plate_img: Plate image
        aspect_threshold: Threshold for 2-line detection

    Returns:
        True if likely 2-line plate
    """
    h, w = plate_img.shape[:2]

    if h == 0:
        return False

    aspect_ratio = w / h

    # Compact layouts are more square
    return aspect_ratio < aspect_threshold


def validate_split_result(line1_text: str, line2_text: str) -> Tuple[bool, str]:
    """
    P0.4 FIX: Re-validate after split to catch errors.

    Rules:
    - Line 1 should match province prefix (2 digits + 1-2 letters)
    - Line 2 should be mostly digits (4-6 digits)

    Returns:
        (is_valid, reason)
    """
    import re

    # Clean texts
    l1 = re.sub(r'[^A-Z0-9]', '', line1_text.upper()) if line1_text else ""
    l2 = re.sub(r'[^A-Z0-9]', '', line2_text.upper()) if line2_text else ""

    # Line 1 validation: Should be province code + series (e.g., "29A", "51G1")
    # Pattern: 2 digits + 1-2 letters (+ optional 1 digit for series like "G1")
    line1_pattern = r'^[0-9]{2}[A-Z]{1,2}[0-9]?$'
    line1_valid = bool(re.match(line1_pattern, l1)) if l1 else False

    # Line 2 validation: Should be mostly/all digits (4-6 chars)
    line2_digit_ratio = sum(1 for c in l2 if c.isdigit()) / len(l2) if l2 else 0
    line2_valid = len(l2) >= 4 and line2_digit_ratio >= 0.8

    if not line1_valid and not line2_valid:
        return False, "Both lines failed validation"
    if not line1_valid:
        return False, f"Line 1 invalid: '{l1}' (expected province+series)"
    if not line2_valid:
        return False, f"Line 2 invalid: '{l2}' (expected digits, got {line2_digit_ratio:.0%} digits)"

    return True, "OK"


def split_and_merge_ocr(plate_img: np.ndarray,
                        ocr_func,
                        fallback_ratio: float = 0.45) -> Tuple[str, float]:
    """
    Split 2-line plate, OCR each line, merge results.
    P0.4 FIX: With post-split validation.

    Args:
        plate_img: Plate image
        ocr_func: OCR function that takes image and returns (text, confidence)
        fallback_ratio: Fallback split ratio

    Returns:
        (merged_text, avg_confidence)
    """
    if not is_2line_plate(plate_img):
        # 1-line plate, OCR directly
        return ocr_func(plate_img)

    # Split adaptively
    line1, line2, _split_row = adaptive_split_2line(plate_img, fallback_ratio)

    # OCR each line
    text1, conf1 = "", 0.0
    text2, conf2 = "", 0.0

    if line1.size > 0:
        try:
            text1, conf1 = ocr_func(line1)
        except Exception as e:
            logger.warning("Line 1 OCR failed: %s", e)

    if line2.size > 0:
        try:
            text2, conf2 = ocr_func(line2)
        except Exception as e:
            logger.warning("Line 2 OCR failed: %s", e)

    # P0.4 FIX: Re-validate split result
    is_valid, reason = validate_split_result(text1, text2)

    if not is_valid:
        # Split failed validation - fallback to single-line OCR
        logger.warning("⚠️ 2-line split failed validation: %s", reason)
        logger.debug("   Falling back to single-line OCR")

        try:
            fallback_text, fallback_conf = ocr_func(plate_img)
            if fallback_text and len(fallback_text) >= 7:
                logger.debug("   Fallback result: '%s' (conf=%.2f)", fallback_text, fallback_conf)
                return fallback_text, fallback_conf
        except Exception as e:
            logger.warning("   Fallback OCR also failed: %s", e)

    # Merge validated results
    merged_text = (text1 or "") + (text2 or "")
    avg_conf = (conf1 + conf2) / 2 if (text1 and text2) else max(conf1, conf2)

    logger.debug("2-line merge: '%s' + '%s' = '%s' (conf=%.2f)", text1, text2, merged_text, avg_conf)

    return merged_text, avg_conf


# Convenience function
def get_adaptive_splitter():
    """Get adaptive line splitter instance."""
    return adaptive_split_2line
