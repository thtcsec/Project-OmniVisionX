"""Vehicle type normalization helpers for omni-vehicle."""

from __future__ import annotations

import re
import unicodedata
from typing import Optional


_MOTORCYCLE_PLATE_PATTERN = re.compile(r"^\d{2}[A-Z]\d{6}$")

_ALIASES = {
    "car": "car",
    "vehicle": "car",
    "auto": "car",
    "automobile": "car",
    "sedan": "car",
    "suv": "car",
    "crossover": "car",
    "hatchback": "car",
    "coupe": "car",
    "mpv": "car",
    "pickup": "car",
    "pickup truck": "car",
    "pickup_truck": "car",
    "pickuptruck": "car",
    "jeep": "car",
    "taxi": "car",
    "cab": "car",
    "minivan": "car",
    "station wagon": "car",
    "wagon": "car",
    "van": "car",
    "delivery van": "car",
    "delivery_van": "car",
    "oto": "car",
    "o to": "car",
    "xe hoi": "car",
    "xe oto": "car",
    "motorcycle": "motorcycle",
    "motorbike": "motorcycle",
    "bike": "motorcycle",
    "motor": "motorcycle",
    "motorcycle rider": "motorcycle",
    "motorcycle-rider": "motorcycle",
    "moto": "motorcycle",
    "scooter": "motorcycle",
    "scooty": "motorcycle",
    "tricycle": "motorcycle",
    "xe may": "motorcycle",
    "xe gan may": "motorcycle",
    "truck": "truck",
    "pickuptrucks": "truck",
    "xe ben": "truck",
    "xe dau keo": "truck",
    "container": "truck",
    "container truck": "truck",
    "tractor truck": "truck",
    "semi": "truck",
    "semi truck": "truck",
    "truck tractor": "truck",
    "xe cong": "truck",
    "lorry": "truck",
    "xe tai": "truck",
    "bus": "bus",
    "coach": "bus",
    "minibus": "bus",
    "xe buyt": "bus",
    "xe buyt nho": "bus",
}

_VEHICLE_HINTS = {
    "car", "vehicle", "auto", "sedan", "suv", "crossover", "hatchback", "coupe",
    "van", "minivan", "mpv", "pickup", "jeep", "taxi", "wagon",
    "truck", "lorry", "container", "semi", "tractor",
    "bus", "coach", "minibus",
    "motorcycle", "motorbike", "moto", "bike", "scooter",
}

_NON_VEHICLE_HINTS = {
    "person", "human", "pedestrian", "face", "head", "bicycle", "cyclist", "animal",
    "dog", "cat", "bag", "helmet", "sign", "traffic light", "trafficlight",
}


def _ascii_key(value: Optional[str]) -> str:
    if not value:
        return ""
    normalized = unicodedata.normalize("NFD", value.strip().lower())
    without_marks = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    cleaned = re.sub(r"[^a-z0-9]+", " ", without_marks)
    return re.sub(r"\s+", " ", cleaned).strip()


def _normalize_plate(plate_number: Optional[str]) -> str:
    if not plate_number:
        return ""
    return re.sub(r"[^A-Z0-9]", "", plate_number.upper())


def infer_vehicle_type_from_plate(plate_number: Optional[str]) -> Optional[str]:
    """
    Infer vehicle type from plate format.
    
    EDGE CASE FIX: If raw plate contains dash/hyphen, it's likely a car plate
    that was OCR'd incorrectly. Don't infer motorcycle in that case.
    """
    if not plate_number:
        return None
    
    # If original plate has dash/hyphen, it's car format (e.g., "51A-12345")
    # Even if OCR removes it → "51A12345", we shouldn't infer motorcycle
    if "-" in plate_number or "/" in plate_number:
        return None
    
    normalized_plate = _normalize_plate(plate_number)
    if _MOTORCYCLE_PLATE_PATTERN.fullmatch(normalized_plate):
        return "motorcycle"
    return None


def normalize_vehicle_type(value: Optional[str], plate_number: Optional[str] = None) -> str:
    key = _ascii_key(value)
    normalized = _ALIASES.get(key, "unknown")

    inferred = infer_vehicle_type_from_plate(plate_number)
    if inferred and normalized in {"unknown", "car"}:
        return inferred

    return normalized


def is_probable_vehicle_label(value: Optional[str]) -> bool:
    key = _ascii_key(value)
    if not key:
        return False
    if key in _ALIASES:
        return True
    if any(token in key for token in _NON_VEHICLE_HINTS):
        return False
    return any(token in key for token in _VEHICLE_HINTS)