"""
Shared plate normalization and validation utilities.
"""
import re
from typing import Set

import cv2
import numpy as np

try:
    from app.services.plate.plate_constants import VALID_PROVINCE_CODES
except Exception:
    VALID_PROVINCE_CODES = None


OCR_TO_DIGIT_MAP = {
    'O': '0', 'Q': '0', 'D': '0',
    'I': '1', 'L': '1',
    'Z': '2',
    'S': '5',
    'B': '8',
    'G': '6',
    'T': '7',
    'A': '4',
}

# Series mapping: I/O are invalid VN series letters, so '1' maps to 'T' and '0' maps to 'D'.
OCR_TO_SERIES_MAP = {
    '8': 'B',
    '5': 'S',
    '2': 'Z',
    '6': 'G',
    '4': 'A',
    '7': 'T',
    '1': 'T',
    '0': 'D',
}


def normalize_plate_basic(text: str) -> str:
    """Normalize plate text to A-Z0-9 only."""
    if not text:
        return ""
    return re.sub(r"[^A-Z0-9]", "", str(text).upper())


def normalize_vn_plate_confusions(text: str) -> str:
    """Normalize common OCR confusion chars by VN plate position rules."""
    clean = normalize_plate_basic(text)
    if not clean:
        return ""

    chars = list(clean)
    raw_len = len(chars)

    for i in range(min(2, len(chars))):
        chars[i] = OCR_TO_DIGIT_MAP.get(chars[i], chars[i])

    if len(chars) >= 3:
        if not chars[2].isalpha():
            chars[2] = OCR_TO_SERIES_MAP.get(chars[2], chars[2])

    serial_start = 3
    if len(chars) >= 4:
        if chars[3].isalpha():
            serial_start = 4
        elif raw_len >= 9 and chars[3] in OCR_TO_SERIES_MAP:
            chars[3] = OCR_TO_SERIES_MAP[chars[3]]
            serial_start = 4

    for i in range(serial_start, len(chars)):
        chars[i] = OCR_TO_DIGIT_MAP.get(chars[i], chars[i])

    return "".join(chars)


def plate_edit_distance(a: str, b: str) -> int:
    """Levenshtein distance — shared implementation for fuzzy plate matching.

    Used by: main.py, event_repository.py, tracking_consensus.py
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    la, lb = len(a), len(b)
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, lb + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[lb]


def fuzzy_plate_match(a: str, b: str, max_dist: int = 2) -> bool:
    """Check if two plate strings are fuzzy-equal (Levenshtein distance ≤ max_dist)."""
    if a == b:
        return True
    if abs(len(a) - len(b)) > max_dist:
        return False
    return plate_edit_distance(a, b) <= max_dist


def get_valid_province_codes() -> Set[str]:
    """Get valid VN province codes as zero-padded strings."""
    if VALID_PROVINCE_CODES:
        return {f"{code:02d}" for code in VALID_PROVINCE_CODES}

    # Fallback list (may be outdated; prefer vn_plate_validator when available)
    # Includes 10 (rare old Hà Nội) — comprehensive per ĐKVN 2024
    return {
        "10",
        "11", "12", "14", "15", "16", "17", "18", "19",
        "20", "21", "22", "23", "24", "25", "26", "27", "28", "29",
        "30", "31", "32", "33", "34", "35", "36", "37", "38", "39",
        "40", "41", "42", "43", "44", "45", "46", "47", "48", "49",
        "50", "51", "52", "53", "54", "55", "56", "57", "58", "59",
        "60", "61", "62", "63", "64", "65", "66", "67", "68", "69",
        "70", "71", "72", "73", "74", "75", "76", "77", "78", "79",
        "80", "81", "82", "83", "84", "85", "86", "87", "88", "89",
        "90", "91", "92", "93", "94", "95", "96", "97", "98", "99",
    }


def is_valid_vn_plate_format(text: str) -> bool:
    """
    Check if text matches Vietnamese plate format.
    Format: XX-Y-NNNNN (province + series + number)
    Examples: 29A12345, 51G99999, 30E12345
    """
    clean = normalize_plate_basic(text)
    if len(clean) < 7 or len(clean) > 10:
        return False

    # First 2 chars must be digits (province code)
    if not clean[:2].isdigit():
        return False

    if clean[:2] not in get_valid_province_codes():
        return False

    # Position 2 must be a letter (series)
    if not clean[2].isalpha():
        return False

    # Valid Vietnamese plate series letters (including J for EV, W for special)
    valid_series = set("ABCDEFGHJKLMNPRSTUVWXYZ")
    if clean[2] not in valid_series:
        return False

    # After series, rest should be digits (allow 2-letter series)
    serial_start = 3
    if len(clean) > 3 and clean[3].isalpha():
        if clean[3] not in valid_series:
            return False
        serial_start = 4

    serial = clean[serial_start:]
    if not serial.isdigit():
        return False

    if len(serial) < 4 or len(serial) > 6:
        return False

    return True


def classify_plate_color_hsv(plate_crop: np.ndarray) -> str:
    if plate_crop is None or not isinstance(plate_crop, np.ndarray) or plate_crop.size == 0:
        return "unknown"

    if len(plate_crop.shape) == 2:
        plate_bgr = cv2.cvtColor(plate_crop, cv2.COLOR_GRAY2BGR)
    elif len(plate_crop.shape) == 3:
        plate_bgr = plate_crop
    else:
        return "unknown"

    hsv = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2HSV)

    h, w = hsv.shape[:2]
    if h < 4 or w < 4:
        return "unknown"

    center = hsv[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    mean_h = float(np.mean(center[:, :, 0]))
    mean_s = float(np.mean(center[:, :, 1]))
    mean_v = float(np.mean(center[:, :, 2]))

    if mean_s < 30 and mean_v > 150:
        return "white"
    if 100 < mean_h < 130 and mean_s > 50:
        return "blue"
    # Red check BEFORE yellow — red plates have H<10 or H>170
    # Yellow check (H<30 and S>100) would wrongly catch H=0-10 red plates
    if mean_h > 170 or mean_h < 10:
        if mean_s > 80:
            return "red"
    if mean_h < 30 and mean_s > 100:
        return "yellow"
    # Green plates (electric vehicles, diplomatic, etc.)
    if 35 < mean_h < 85 and mean_s > 50:
        return "green"

    return "unknown"
