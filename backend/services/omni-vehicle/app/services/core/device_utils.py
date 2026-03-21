import logging
import os
from typing import Optional

import torch

logger = logging.getLogger("omni-vehicle.device")

_cuda_usable_cache: Optional[bool] = None


def cuda_is_usable() -> bool:
    """Return True only if CUDA is both available and usable for kernels."""
    global _cuda_usable_cache
    if _cuda_usable_cache is not None:
        return _cuda_usable_cache

    force_cpu = os.environ.get("FORCE_CPU", "").strip().lower() in {"1", "true", "yes"}
    if force_cpu:
        _cuda_usable_cache = False
        return False

    try:
        if not torch.cuda.is_available():
            _cuda_usable_cache = False
            return False

        # Try a tiny CUDA op to ensure kernels can run on this GPU
        _ = torch.tensor([1.0], device="cuda") + 1.0
        torch.cuda.synchronize()
        _cuda_usable_cache = True
        return True
    except Exception as exc:
        logger.warning("CUDA not usable, falling back to CPU: %s", exc)
        _cuda_usable_cache = False
        return False


def resolve_device(preferred: str = "cuda") -> str:
    """Resolve preferred device to a safe string ('cuda' or 'cpu')."""
    if preferred.lower().startswith("cuda") and cuda_is_usable():
        return "cuda:0"
    return "cpu"


def get_torch_device(preferred: str = "cuda") -> torch.device:
    """Get a torch.device honoring CUDA usability checks."""
    return torch.device(resolve_device(preferred))
