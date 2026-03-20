"""
AmbientAdapter — Auto Day/Night Threshold Interpolation (v2)
==============================================================
Production-grade per-camera ambient brightness tracker with auto
threshold interpolation between pre-tuned night/day values.

Key design decisions (v2 — Phase 6 optimization):
  • Profile night/day RATIOS are authoritative. The operator's slider_value
    is used as the DAY anchor, and the night endpoint is derived from the
    profile ratio.  This means UI sliders actually work while maintaining
    the tested night-relaxation proportions.
  • Per-camera adaptive floor/ceil via rolling min/max EMA replaces
    fixed _DARK_FLOOR / _BRIGHT_CEIL, handling WDR/gain/IR differences.
  • EMA alpha is time-based (target τ = 2s) so behavior is consistent
    regardless of actual frame-rate (2 FPS LPR vs 30 FPS preview).
  • Hysteresis deadzone (±5 %) eliminates ratio flicker near boundaries.
  • Background eviction daemon thread prevents unbounded dict growth.
  • Transition logging: one debug line when camera crosses night/day.

Usage:
    adapter = AmbientAdapter.get_instance()
    adapter.update_brightness(camera_id, frame_bgr)
    eff = adapter.get_threshold(camera_id, "ocr_confidence_threshold",
                                settings.ocr_confidence_threshold)
"""
from __future__ import annotations

import atexit
import logging
import threading
import time
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger("omni-vehicle.ambient")

# ── Day/Night endpoint values for each tunable threshold ──────
# Format: {setting_name: (night_value, day_value)}
#
# ambient_ratio = 0.0 → full night (dark frame)
# ambient_ratio = 1.0 → full day (bright frame)
# effective = night_val + ambient_ratio * (day_val - night_val)
#
# NOTE: slider_value is used as the DAY anchor. The night value is
# derived proportionally: night_effective = slider * (night_val/day_val).
# This preserves tested night/day ratios while letting operators tune.
_THRESHOLD_PROFILES: Dict[str, Tuple[float, float]] = {
    # Detection thresholds — night: lower for recall, day: higher for precision
    "fortress_vehicle_confidence": (0.10, 0.20),
    "fortress_plate_confidence":   (0.15, 0.30),
    "plate_detector_confidence":   (0.15, 0.30),
    "ocr_confidence_threshold":    (0.35, 0.50),
    "event_instant_confidence":    (0.72, 0.85),
    # Adaptive confidence
    "adaptive_confidence_base":    (0.32, 0.45),
    "adaptive_confidence_alpha":   (0.18, 0.12),
    "adaptive_confidence_min":     (0.15, 0.25),
}

# Integer thresholds (consensus/vote — interpolated then rounded)
_INT_PROFILES: Dict[str, Tuple[int, int]] = {
    "lpr_consensus_history": (15, 8),    # night: more votes, day: fewer
    "event_min_vote_count":  (3, 2),     # night: need more agreement
}

# ── Constants ─────────────────────────────────────────────────
# Initial floor/ceil (only used for the very first frames before
# per-camera adaptive range kicks in).  These are intentionally
# conservative — the adaptive range will overwrite them within
# ~30 seconds of real data.
_INITIAL_DARK_FLOOR = 40.0
_INITIAL_BRIGHT_CEIL = 150.0

# Adaptive range EMA — slow (α=0.002 ≈ 500 updates to converge)
# so transient headlights don't blow out the range.
_RANGE_EMA_ALPHA = 0.002

# Minimum span between floor and ceil to avoid division-by-zero
# or hyper-sensitivity when a camera is in a near-constant lit scene.
_MIN_RANGE_SPAN = 30.0

# Time constant for brightness EMA (seconds).
# At any frame-rate the effective alpha = 1 - exp(-Δt / τ).
_EMA_TAU = 2.0  # 2 seconds ≈ 0.05 alpha at 30 FPS, 0.39 at 2 FPS

# Hysteresis deadzone: ratios below this → clamp 0, above (1-this) → clamp 1.
_DEADZONE = 0.05

_STALE_TTL = 120.0     # Drop camera entry after 2 min no update
_EVICT_INTERVAL = 30.0  # Background eviction check interval

# Transition thresholds for logging (avoid log spam)
_NIGHT_THRESHOLD = 0.20
_DAY_THRESHOLD = 0.80


class _CameraState:
    """Per-camera mutable state.  All fields guarded by AmbientAdapter._cam_lock."""
    __slots__ = (
        "brightness_ema", "last_update", "ambient_ratio",
        "range_floor_ema", "range_ceil_ema",
        "prev_mode",  # "night" | "day" | "transition" — for logging
    )

    def __init__(self) -> None:
        self.brightness_ema: float = 100.0              # neutral start
        self.last_update: float = 0.0
        self.ambient_ratio: float = 0.5                 # neutral start
        self.range_floor_ema: float = _INITIAL_DARK_FLOOR
        self.range_ceil_ema: float = _INITIAL_BRIGHT_CEIL
        self.prev_mode: str = "transition"


def _compute_effective_alpha(dt: float) -> float:
    """Time-based EMA alpha: consistent behavior regardless of FPS.

    α = 1 - exp(-Δt / τ)  where τ = _EMA_TAU.
    Falls back to 0.05 for the very first frame (dt=0).
    """
    if dt <= 0.0:
        return 0.05
    import math
    return 1.0 - math.exp(-dt / _EMA_TAU)


class AmbientAdapter:
    """Per-camera brightness tracker with auto threshold interpolation."""

    _instance: Optional[AmbientAdapter] = None
    _lock = threading.Lock()
    # Thread-local storage so executor threads can access current camera_id
    # without changing fortress_lpr method signatures.
    _tls = threading.local()

    def __init__(self) -> None:
        self._cameras: Dict[str, _CameraState] = {}
        self._cam_lock = threading.Lock()
        # Background eviction daemon
        self._evict_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._start_eviction_daemon()

    def _start_eviction_daemon(self) -> None:
        """Launch a daemon thread that periodically evicts stale cameras."""
        if self._evict_thread is not None and self._evict_thread.is_alive():
            return
        self._stop_event.clear()
        t = threading.Thread(target=self._eviction_loop, daemon=True,
                             name="ambient-evict")
        t.start()
        self._evict_thread = t
        # Ensure clean shutdown
        atexit.register(self._stop_eviction_daemon)

    def _stop_eviction_daemon(self) -> None:
        self._stop_event.set()

    def _eviction_loop(self) -> None:
        """Background loop: every _EVICT_INTERVAL seconds, drop stale cameras."""
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=_EVICT_INTERVAL)
            if self._stop_event.is_set():
                break
            try:
                n = self.evict_stale()
                if n > 0:
                    logger.debug("Background eviction: removed %d stale camera(s)", n)
            except Exception:
                pass  # never crash the daemon

    @classmethod
    def get_instance(cls) -> AmbientAdapter:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = AmbientAdapter()
        return cls._instance

    # ── Thread-local camera context ───────────────────────────
    # Call set_active_camera() in the executor thread before running
    # fortress pipeline methods.  fortress_lpr.py reads it back via
    # get_active_camera() to look up per-camera ambient thresholds.

    @classmethod
    def set_active_camera(cls, camera_id: str) -> None:
        cls._tls.camera_id = camera_id

    @classmethod
    def get_active_camera(cls) -> str:
        return getattr(cls._tls, "camera_id", "unknown")

    # ── Core API ──────────────────────────────────────────────

    def update_brightness(self, camera_id: str, frame_bgr: np.ndarray,
                          plate_bbox: Optional[Tuple[int, int, int, int]] = None,
                          ) -> float:
        """Feed a frame to update the per-camera brightness EMA.

        Call this once per processed frame (inside _process_detection).
        Returns the current ambient_ratio [0=night, 1=day].

        Args:
            camera_id: Camera identifier.
            frame_bgr: Full frame in BGR format.
            plate_bbox: Optional (x1, y1, x2, y2) of detected plate/vehicle.
                        If provided, brightness is measured from this region
                        (more relevant than center ROI). Falls back to center
                        25% + weighted lower half when not available.
        """
        # Measure brightness: V-channel 90th percentile
        try:
            if frame_bgr.ndim == 3:
                h, w = frame_bgr.shape[:2]
                if plate_bbox is not None:
                    # Priority: use plate/vehicle bounding box region
                    bx1, by1, bx2, by2 = plate_bbox
                    bx1 = max(0, min(bx1, w - 1))
                    by1 = max(0, min(by1, h - 1))
                    bx2 = max(bx1 + 1, min(bx2, w))
                    by2 = max(by1 + 1, min(by2, h))
                    roi = frame_bgr[by1:by2, bx1:bx2]
                else:
                    # Weighted center 25% + lower-center (road is usually darker
                    # than sky, lower half is more representative of plate area)
                    cy0, cy1 = h // 4, h * 3 // 4
                    cx0, cx1 = w // 4, w * 3 // 4
                    center_roi = frame_bgr[cy0:cy1, cx0:cx1]
                    # Lower-center region (bottom 25%, center 50% width)
                    ly0 = h * 3 // 4
                    lx0, lx1 = w // 4, w * 3 // 4
                    lower_roi = frame_bgr[ly0:h, lx0:lx1]
                    # Combine with 60% weight on center, 40% on lower
                    roi = center_roi  # primary
                    # We'll measure both and blend
                    v_center = np.max(center_roi, axis=2)
                    v_lower = np.max(lower_roi, axis=2)
                    p90_center = float(np.percentile(v_center, 90))
                    p90_lower = float(np.percentile(v_lower, 90))
                    brightness = 0.6 * p90_center + 0.4 * p90_lower
                    roi = None  # signal that brightness is already computed

                if roi is not None:
                    # V = max(B, G, R) per pixel — cheaper than full cvtColor
                    v_channel = np.max(roi, axis=2)
                    brightness = float(np.percentile(v_channel, 90))
            else:
                brightness = float(np.percentile(frame_bgr, 90))
        except Exception:
            brightness = 100.0

        now = time.monotonic()

        with self._cam_lock:
            state = self._cameras.get(camera_id)
            if state is None:
                state = _CameraState()
                self._cameras[camera_id] = state

            # Time-based EMA alpha — consistent across any frame rate
            dt = now - state.last_update if state.last_update > 0 else 0.0
            alpha = _compute_effective_alpha(dt)

            # EMA update for brightness
            if state.last_update == 0.0:
                state.brightness_ema = brightness
            else:
                state.brightness_ema += alpha * (brightness - state.brightness_ema)

            # Adaptive floor/ceil per camera (slow EMA)
            # Track rolling min via EMA towards brightness when brightness < floor
            # Track rolling max via EMA towards brightness when brightness > ceil
            if brightness < state.range_floor_ema:
                state.range_floor_ema += _RANGE_EMA_ALPHA * (brightness - state.range_floor_ema)
            if brightness > state.range_ceil_ema:
                state.range_ceil_ema += _RANGE_EMA_ALPHA * (brightness - state.range_ceil_ema)
            # Also slowly relax towards the current brightness to avoid stuck range
            # (e.g., camera was reconfigured)
            state.range_floor_ema += _RANGE_EMA_ALPHA * 0.1 * (brightness - state.range_floor_ema)
            state.range_ceil_ema += _RANGE_EMA_ALPHA * 0.1 * (brightness - state.range_ceil_ema)

            # Enforce minimum span
            span = state.range_ceil_ema - state.range_floor_ema
            if span < _MIN_RANGE_SPAN:
                mid = (state.range_ceil_ema + state.range_floor_ema) / 2.0
                state.range_floor_ema = mid - _MIN_RANGE_SPAN / 2.0
                state.range_ceil_ema = mid + _MIN_RANGE_SPAN / 2.0

            state.last_update = now

            # Compute ambient ratio with per-camera adaptive range
            floor = state.range_floor_ema
            ceil = state.range_ceil_ema
            raw_ratio = (state.brightness_ema - floor) / (ceil - floor)
            ratio = max(0.0, min(1.0, raw_ratio))

            # Hysteresis deadzone to eliminate flicker
            if ratio < _DEADZONE:
                ratio = 0.0
            elif ratio > (1.0 - _DEADZONE):
                ratio = 1.0

            state.ambient_ratio = ratio

            # Log ambient transitions (once per change, not every frame)
            new_mode = (
                "night" if ratio < _NIGHT_THRESHOLD else
                "day" if ratio > _DAY_THRESHOLD else
                "transition"
            )
            if new_mode != state.prev_mode:
                logger.info(
                    "Ambient transition [%s]: %s → %s  "
                    "(ratio=%.3f, ema=%.1f, floor=%.1f, ceil=%.1f)",
                    camera_id, state.prev_mode, new_mode,
                    ratio, state.brightness_ema,
                    state.range_floor_ema, state.range_ceil_ema,
                )
                state.prev_mode = new_mode

            return state.ambient_ratio

    def get_ambient_ratio(self, camera_id: str) -> float:
        """Get the current ambient ratio for a camera.
        Returns 0.5 (neutral) if camera not yet tracked.
        """
        with self._cam_lock:
            state = self._cameras.get(camera_id)
            return state.ambient_ratio if state else 0.5

    def get_threshold(self, camera_id: str, setting_name: str,
                      slider_value: float) -> float:
        """Get the effective threshold for a setting, interpolated by ambient.

        Profile night/day RATIOS are preserved (i.e. "night is 70% of day"),
        but the user's slider_value is used as the DAY endpoint. This way
        operators can tune thresholds via UI while still getting automatic
        night relaxation.

        Example: ocr_confidence_threshold profile is (0.35, 0.50).
          Profile ratio = 0.35/0.50 = 0.70 (night is 70% of day).
          User sets slider to 0.40 → effective_night = 0.40*0.70 = 0.28.
          At ambient_ratio=0.5: 0.28 + 0.5*(0.40-0.28) = 0.34.

        Args:
            camera_id: Camera identifier
            setting_name: Name matching a key in _THRESHOLD_PROFILES
            slider_value: Current value from Settings (used as day anchor)

        Returns:
            Effective threshold value.
        """
        ratio = self.get_ambient_ratio(camera_id)

        if setting_name in _THRESHOLD_PROFILES:
            night_val, day_val = _THRESHOLD_PROFILES[setting_name]
            # Use slider_value as the day endpoint; derive night proportionally
            if day_val > 0:
                night_factor = night_val / day_val  # e.g. 0.70 for ocr
                effective_night = slider_value * night_factor
            else:
                effective_night = night_val
            return effective_night + ratio * (slider_value - effective_night)

        if setting_name in _INT_PROFILES:
            night_val, day_val = _INT_PROFILES[setting_name]
            if day_val > 0:
                night_factor = night_val / day_val
                effective_night = slider_value * night_factor
            else:
                effective_night = night_val
            return round(effective_night + ratio * (slider_value - effective_night))

        return slider_value

    def get_int_threshold(self, camera_id: str, setting_name: str,
                          slider_value: int) -> int:
        """Convenience: get_threshold but always returns int."""
        return int(round(self.get_threshold(camera_id, setting_name, float(slider_value))))

    # ── Maintenance ───────────────────────────────────────────

    def evict_stale(self) -> int:
        """Remove cameras that haven't sent frames in > _STALE_TTL seconds."""
        now = time.monotonic()
        with self._cam_lock:
            stale = [
                cam_id for cam_id, state in self._cameras.items()
                if (now - state.last_update) > _STALE_TTL
            ]
            for cam_id in stale:
                del self._cameras[cam_id]
            if stale:
                logger.debug(
                    "Evicted %d stale ambient camera(s): %s",
                    len(stale), ", ".join(stale),
                )
            return len(stale)

    def get_all_states(self) -> Dict[str, dict]:
        """Return snapshot of all camera ambient states (for /stats API & metrics)."""
        with self._cam_lock:
            return {
                cam_id: {
                    "brightness_ema": round(state.brightness_ema, 1),
                    "ambient_ratio": round(state.ambient_ratio, 3),
                    "range_floor": round(state.range_floor_ema, 1),
                    "range_ceil": round(state.range_ceil_ema, 1),
                    "mode": state.prev_mode,
                    "last_update_age": round(time.monotonic() - state.last_update, 1),
                }
                for cam_id, state in self._cameras.items()
            }
