"""
Custom exceptions for omni-vehicle pipeline.
Enables specific error handling and clearer telemetry/debugging.
"""


class LprBaseError(Exception):
    """Base exception for LPR pipeline."""

    pass


class ModelLoadError(LprBaseError):
    """Raised when a model (YOLO, Fortress, etc.) fails to load."""

    pass


class PlateRecognitionError(LprBaseError):
    """Raised when plate detection or OCR fails with a known cause."""

    pass


class PlateDetectionEmptyError(PlateRecognitionError):
    """Raised when no plate was detected (empty crop, blur, etc.)."""

    pass
