"""
Shared constants for VN license plate validation.
Single source of truth — all ALPR components import from here.

References:
- Thông tư 58/2020/TT-BCA (plate format regulations)
- Updated for 2025-2026 province mergers
"""
from typing import Dict, List, Set


# ══════════════════════════════════════════════════════════════
# Province Codes (2026)
# ══════════════════════════════════════════════════════════════
# Includes ALL historical codes (pre-merger vehicles still on the road)
# plus new codes from 2025 province mergers.
# Some codes (13, 44-49, 52-59, etc.) were never officially assigned
# but we keep broad coverage to avoid rejecting edge cases.
VALID_PROVINCE_CODES: Set[int] = {
    11, 12, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
    30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
    40, 41, 43,
    47, 48, 49,
    50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
    60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
    70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 83, 84, 85, 86, 88, 89,
    90, 91, 92, 93, 94, 95, 97, 98, 99,
}

# Multi-prefix provinces (one province → many codes)
PROVINCE_MULTI_PREFIX: Dict[str, List[int]] = {
    "Hà Nội":      [29, 30, 31, 32, 33, 40],
    "TP.HCM":      [41, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
    "Hải Phòng":   [15, 16],
    "Đồng Nai":    [39, 60],
    "An Giang":    [67, 68],
}


# ══════════════════════════════════════════════════════════════
# Series Letters
# ══════════════════════════════════════════════════════════════
# I, O, Q excluded (OCR confusion with 1, 0, 0).
# J included (EV plates). W included (special/rare but valid).
VALID_SERIES_LETTERS: Set[str] = set("ABCDEFGHJKLMNPRSTUVWXYZ")


# ══════════════════════════════════════════════════════════════
# Serial Digit Spec
# ══════════════════════════════════════════════════════════════
# Standard: 5 digits (000.01 – 999.99)
# OCR tolerance: accept 4-6 with confidence penalty
SERIAL_DIGIT_LENGTH = 5
SERIAL_DIGIT_MIN = 4    # OCR noise tolerance
SERIAL_DIGIT_MAX = 6    # OCR noise tolerance


# ══════════════════════════════════════════════════════════════
# Province Digit Confusion Graph
# ══════════════════════════════════════════════════════════════
# Maps each digit to visually similar digits in OCR/IR.
# Used for province correction instead of naive abs() ±1.
DIGIT_CONFUSION_GRAPH: Dict[str, List[str]] = {
    '0': ['8', '6', 'D', 'O', 'Q'],
    '1': ['7', 'I', 'L', 'T'],
    '2': ['7', 'Z'],
    '3': ['8'],
    '4': ['A'],
    '5': ['6', '8', 'S'],
    '6': ['8', '0', '5', 'G'],
    '7': ['1', '2', 'T'],
    '8': ['0', '6', '3', 'B'],
    '9': ['8', 'g', 'q'],
}
