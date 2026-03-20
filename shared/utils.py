"""
Shared Utilities for OmniVision Services
"""
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def get_env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def get_env_bool(key: str, default: bool = False) -> bool:
    val = os.environ.get(key, str(default)).lower()
    return val in ('true', '1', 'yes', 'on')


def get_env_int(key: str, default: int = 0) -> int:
    try:
        return int(os.environ.get(key, default))
    except (ValueError, TypeError):
        return default


def verify_internal_secret(secret: Optional[str], configured: str) -> bool:
    """Verify internal API secret for service-to-service calls"""
    if not configured:
        return True  # No secret configured
    return secret == configured
