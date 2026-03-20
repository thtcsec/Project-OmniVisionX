"""
License Plate Recognition (LPR) Service
Wraps PaddleOCR with optimizations for:
- Lazy Loading (Save memory on startup)
- Bbox Expansion (Prevent tight crops)
- Weighted Voting (Temporal Consensus with Best-Shot)
- VN Plate Normalization (Fix OCR errors)
"""
import asyncio
import logging
import os
import re
import time
import threading
from collections import Counter
from typing import Optional, List, Dict, Tuple
from PIL import Image
import numpy as np

# Import only typing here to avoid early heavy import
try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None

from app.config import get_settings
from app.services.core.enhancer import ImageEnhancer
from app.services.plate.plate_utils import normalize_plate_basic, normalize_vn_plate_confusions, is_valid_vn_plate_format
from app.services.core.image_source_detector import estimate_sharpness
try:
    from app.services.plate.vn_plate_validator import validate_and_correct_plate
    VN_VALIDATOR_AVAILABLE = True
except Exception:
    VN_VALIDATOR_AVAILABLE = False

logger = logging.getLogger(__name__)

# ── Sharpness-based quality penalty ──────────────────────────
# Floor: minimum multiplier even for perfectly blurry plates.
# Range: additional multiplier earned by sharp plates (floor + range = 1.0).
# Normalizer: sharpness value at which the plate is considered "sharp enough".
SHARPNESS_PENALTY_FLOOR = 0.85
SHARPNESS_PENALTY_RANGE = 0.15
SHARPNESS_NORMALIZER = 200.0

# ── GPU recovery ─────────────────────────────────────────────
# Delay (seconds) before attempting to reclaim GPU after a transient failure.
GPU_RECOVERY_DELAY_S = 60


def expand_bbox(x1: float, y1: float, x2: float, y2: float,
                img_w: int, img_h: int, scale: float = 0.2) -> Tuple[int, int, int, int]:
    """DEPRECATED — use lpr_utils.expand_bbox instead. Kept for import compat."""
    from app.services.pipeline.application.lpr_utils import expand_bbox as _expand
    return _expand(int(x1), int(y1), int(x2), int(y2), img_w, img_h, scale)


def expand_plate_box(box: list, img_w: int, img_h: int, scale: float = 0.3) -> Tuple[int, int, int, int]:
    """
    Expand PaddleOCR text line box to include plate borders.
    PaddleOCR box is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] - just the text, not plate border.

    Args:
        box: PaddleOCR polygon box (4 points)
        img_w, img_h: Image dimensions for clamping
        scale: Expansion ratio (0.3 day, 0.45 night)

    Returns:
        Expanded bbox as (x1, y1, x2, y2) integers with validation
    """
    try:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]

        x1, y1 = min(xs), min(ys)
        x2, y2 = max(xs), max(ys)

        bw = x2 - x1
        bh = y2 - y1

        # Validate non-zero dimensions
        if bw <= 0 or bh <= 0:
            return (0, 0, img_w, img_h)  # Return full image as fallback

        dx = int(bw * scale)
        dy = int(bh * scale)

        nx1 = max(0, int(x1 - dx))
        ny1 = max(0, int(y1 - dy))
        nx2 = min(img_w, int(x2 + dx))
        ny2 = min(img_h, int(y2 + dy))

        # Final validation: ensure x2 > x1 and y2 > y1
        if nx2 <= nx1 or ny2 <= ny1:
            return (0, 0, img_w, img_h)

        return nx1, ny1, nx2, ny2
    except (IndexError, TypeError):
        return (0, 0, img_w, img_h)


def normalize_vn_plate(text: str) -> str:
    return normalize_vn_plate_confusions(text)


class OCRService:
    _instance = None
    _instance_lock = threading.Lock()
    _ocr_model = None
    _external_model = None

    # Config - loaded from env
    MAX_VOTE_HISTORY = 10  # Keep last 10 frames
    VOTE_BUFFER_TTL = 60.0  # Seconds before stale track cleanup
    VOTE_CLEANUP_INTERVAL = 10.0  # Seconds between cleanup sweeps
    VOTE_ENTRY_TTL = 10.0  # Per-entry freshness window (10s covers traffic jams + slow vehicles)

    # Rate Limits
    MAX_SYSTEM_FPS = 60  # Increased for multiple cameras
    MAX_CAMERA_FPS = 15  # Faster OCR for moving vehicles

    def __init__(self):
        self.enhancer = ImageEnhancer()
        self._last_call_time = {}  # {camera_id: timestamp}
        self._system_call_count = 0
        self._system_call_window = 0
        self._lock = asyncio.Lock()  # Thread-safe rate limiter lock
        self._ocr_lock = threading.Lock()  # Serialize PaddleOCR calls (not thread-safe)
        self._vote_lock = threading.Lock()  # Serialize voting buffer access

        # Instance-level voting buffer (not class-level to avoid shared state)
        self._voting_buffer: Dict[int, List[Dict]] = {}
        self._last_cleanup_time: float = 0.0

    @property
    def CONFIDENCE_THRESHOLD(self) -> float:
        """Read-through from settings singleton — live-updated by FE sliders."""
        env_conf = os.environ.get("OCR_CONF_THRESHOLD")
        if env_conf is not None:
            return float(env_conf)
        return float(get_settings().ocr_confidence_threshold)

    @property
    def MIN_VOTE_COUNT(self) -> int:
        """Read-through from settings singleton — live-updated by FE sliders."""
        env_votes = os.environ.get("EVENT_MIN_VOTE_COUNT")
        if env_votes is not None:
            return int(env_votes)
        return int(get_settings().event_min_vote_count)

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = OCRService()
        return cls._instance

    def set_external_model(self, model) -> None:
        if model is not None:
            self._external_model = model
            self._ocr_model = model

    def _get_model(self):
        """
        Lazy Load PaddleOCR Model.
        PaddleOCR 2.9.x API - use use_gpu=True for GPU acceleration.
        NOTE: PaddleOCR 3.0 has bugs with set_optimization_level
        """
        if self._external_model is not None:
            return self._external_model

        if self._ocr_model is None:
            if PaddleOCR is None:
                logger.error("PaddleOCR not installed!")
                raise RuntimeError("PaddleOCR not installed")

            logger.info("Initializing PaddleOCR (Lazy Load)...")

            # Suppress verbose logs from PaddlePaddle
            os.environ["FLAGS_log_level"] = "3"
            os.environ["GLOG_minloglevel"] = "3"

            # Check if cuDNN is actually available before trying GPU
            # PaddlePaddle can fail at runtime even if CUDA works
            use_gpu = False
            try:
                import paddle
                if paddle.is_compiled_with_cuda():
                    # Test if cuDNN actually works
                    try:
                        gpu_count = paddle.device.cuda.device_count()
                        if gpu_count > 0:
                            use_gpu = True
                            logger.info("✅ CUDA + cuDNN available for PaddleOCR (devices=%d)", gpu_count)
                        else:
                            logger.warning("⚠️ CUDA compiled but no GPU devices (count=%d)", gpu_count)
                    except Exception as cudnn_err:
                        logger.warning("cuDNN not available: %s", cudnn_err)
                        use_gpu = False
            except ImportError:
                use_gpu = False

            # PaddleOCR API - GPU first with fallback to CPU
            # use_angle_cls=False: plate crops are already oriented from detection;
            # angle classification wastes ~5-10ms per plate with no benefit.
            try:
                self._ocr_model = PaddleOCR(
                    lang='en',
                    use_gpu=use_gpu,
                    show_log=False,
                    use_angle_cls=False,
                    det_db_box_thresh=0.3
                )
                logger.info("✅ PaddleOCR Initialized (%s)", 'GPU' if use_gpu else 'CPU')
            except Exception as e:
                logger.warning("PaddleOCR init failed: %s, trying CPU...", e)
                try:
                    self._ocr_model = PaddleOCR(
                        lang='en',
                        use_gpu=False,
                        show_log=False,
                        use_angle_cls=False,
                        det_db_box_thresh=0.3
                    )
                    logger.info("✅ PaddleOCR Initialized (CPU fallback)")
                except Exception as e2:
                    logger.error("Failed to init PaddleOCR: %s", e2)
                    self._ocr_model = None

        return self._ocr_model

    def _reset_model_to_cpu(self):
        """Force reset model to CPU mode after GPU failure.

        Schedules a GPU recovery attempt after 60 seconds so the system
        doesn't stay on slow CPU forever after a transient GPU error.
        """
        logger.warning("🔄 Resetting PaddleOCR to CPU mode due to GPU error...")
        self._external_model = None
        self._ocr_model = None
        try:
            self._ocr_model = PaddleOCR(
                lang='en',
                use_gpu=False,
                show_log=False,
                use_angle_cls=False,
                det_db_box_thresh=0.3
            )
            logger.info("✅ PaddleOCR reset to CPU mode")
            # Schedule GPU recovery attempt
            self._schedule_gpu_recovery()
            return self._ocr_model
        except Exception as e:
            logger.error("Failed to reset PaddleOCR to CPU: %s", e)
            return None

    def _schedule_gpu_recovery(self):
        """Try to recover GPU after a delay (non-blocking)."""
        def _try_gpu():
            time.sleep(GPU_RECOVERY_DELAY_S)
            try:
                import paddle
                if not paddle.is_compiled_with_cuda():
                    return
                gpu_count = paddle.device.cuda.device_count()
                if gpu_count <= 0:
                    return
                test_model = PaddleOCR(
                    lang='en', use_gpu=True, show_log=False,
                    use_angle_cls=False, det_db_box_thresh=0.3
                )
                # If we get here, GPU works again
                with self._ocr_lock:
                    self._external_model = None
                    self._ocr_model = test_model
                logger.info("✅ PaddleOCR GPU recovered successfully")
            except Exception as e:
                logger.debug("GPU recovery failed (staying on CPU): %s", e)

        t = threading.Thread(target=_try_gpu, daemon=True, name="paddleocr-gpu-recovery")
        t.start()

    async def _check_rate_limit(self, camera_id: str) -> bool:
        """
        Hard Rate Limiter (Thread-safe).
        Returns True if allowed, False if dropped.
        """
        now = time.time()

        async with self._lock:
            # 1. System Limit (FPS check)
            if now - self._system_call_window > 1.0:
                self._system_call_window = now
                self._system_call_count = 0

            if self._system_call_count >= self.MAX_SYSTEM_FPS:
                return False

            # 2. Camera Limit
            last = self._last_call_time.get(camera_id, 0)
            if now - last < (1.0 / self.MAX_CAMERA_FPS):
                return False

            # Update counters
            self._system_call_count += 1
            self._last_call_time[camera_id] = now
            return True

    async def predict_plate(
        self,
        plate_image: Image.Image,
        track_id: Optional[int] = None,
        camera_id: Optional[str] = None,
        bbox_area: int = 0,
        is_night: bool = False
    ) -> Optional[Tuple[str, float, list]]:
        """
        Run OCR on a license plate image with dual preprocessing fallback.
        Applies preprocessing, temporal voting if track_id provided,
        and Vietnamese plate normalization.

        Args:
            plate_image: Cropped vehicle/plate image
            track_id: Tracking ID for temporal voting
            camera_id: Camera ID for rate limiting
            bbox_area: Bbox area for best-shot selection in voting
            is_night: Night mode flag for processing adjustments

        Returns:
            Tuple of (plate_text, confidence, box) or None if not ready
        """
        # Rate Limit Check
        if camera_id and not await self._check_rate_limit(camera_id):
            return None

        try:
            loop = asyncio.get_running_loop()

            # 1. Smart preprocessing based on day/night mode
            if is_night:
                enhanced_img = self.enhancer.preprocess_for_ocr(plate_image, use_threshold=False)
            else:
                enhanced_img = self.enhancer.preprocess_smart(plate_image, is_night=is_night)

            # 2. Run OCR on enhanced image
            result = await loop.run_in_executor(None, self._run_ocr_sync, enhanced_img)

            text, score, box = None, 0.0, None

            if result:
                text, score, box = result

            # 3. If score is low, try alternative preprocessing
            if score < 0.5:
                if is_night:
                    # Night: try specialized night preprocessing
                    alt_img = self.enhancer.preprocess_night_plate(plate_image)
                else:
                    alt_img = self.enhancer.preprocess_for_ocr(plate_image, use_threshold=True)
                alt_result = await loop.run_in_executor(None, self._run_ocr_sync, alt_img)
                if alt_result:
                    alt_text, alt_score, alt_box = alt_result
                    if alt_score > score:
                        text, score, box = alt_text, alt_score, alt_box

            if not text or score < self.CONFIDENCE_THRESHOLD:
                return None

            # 4. Apply Vietnamese plate normalization (fix O->0, I->1, etc.)
            text = normalize_vn_plate(text)

            sharpness = 0.0
            quality_factor = 1.0  # Neutral default if estimation fails
            try:
                sharpness = float(estimate_sharpness(np.array(plate_image)))
                # Reduced penalty: range 0.85-1.0 instead of old 0.7-1.0.
                # Old penalty was too harsh for distant vehicles (sharpness~10
                # would tank a 0.80 score to 0.56, below OCR threshold).
                quality_factor = SHARPNESS_PENALTY_FLOOR + min(sharpness / SHARPNESS_NORMALIZER, 1.0) * SHARPNESS_PENALTY_RANGE
            except Exception:
                pass
            score = min(score * quality_factor, 0.99)

            if VN_VALIDATOR_AVAILABLE:
                try:
                    result = validate_and_correct_plate(text, score)
                    text = result.corrected
                    score = max(0.0, score - result.confidence_penalty)
                except Exception:
                    pass

            format_valid = is_valid_vn_plate_format(text)
            if format_valid:
                score = min(score + 0.05, 0.99)
            else:
                score *= 0.85

            # 5. Temporal Voting (use area for best-shot)
            valid_area = max(bbox_area, 1)

            # HIGH CONFIDENCE: Return immediately but still feed voting buffer
            # so subsequent frames can benefit from consensus history
            if score >= 0.75 and format_valid:
                if track_id is not None:
                    self._apply_voting(track_id, text, score, area=valid_area)
                return (text, score, box)

            # MEDIUM CONFIDENCE: Use voting if track_id available
            if track_id is not None:
                # Allow marginal plates (0.35+) into voting buffer to build consensus
                if score >= 0.35:
                    vote_result = self._apply_voting(track_id, text, score, area=valid_area)
                    if vote_result:
                        final_text, consensus_score = vote_result
                        # Use the consensus winner's best score, not the
                        # current frame's score, to prevent mismatched
                        # text/confidence that causes downstream filtering.
                        return (final_text, consensus_score, box)
                return None

            # LOW CONFIDENCE without tracking: strictly fallback to threshold
            if score >= self.CONFIDENCE_THRESHOLD and format_valid:
                return (text, score, box)

            return None

        except Exception as e:
            logger.error("OCR Prediction failed: %s", e)
            return None

    def _run_ocr_sync(self, image: Image.Image):
        """Internal synchronous OCR call."""
        # Validate image size - need minimum dimensions
        if image.width < 20 or image.height < 10:
            logger.debug("Image too small: %sx%s", image.width, image.height)
            return None

        # Ensure RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')

        img_np = np.array(image)

        # Verify shape is valid
        if len(img_np.shape) != 3 or img_np.shape[2] != 3:
            logger.debug("Invalid image shape: %s", img_np.shape)
            return None

        model = self._get_model()

        try:
            # PaddleOCR 2.x stable API: ocr()
            # Serialize: PaddleOCR is NOT thread-safe (internal state shared)
            with self._ocr_lock:
                result = model.ocr(img_np, cls=True)
        except Exception as e:
            error_str = str(e)
            # Check for cuDNN/GPU errors and fallback to CPU
            if 'cudnn' in error_str.lower() or 'cuda' in error_str.lower():
                logger.warning("PaddleOCR GPU error: %s, switching to CPU...", e)
                model = self._reset_model_to_cpu()
                if model is None:
                    return None
                # Retry with CPU
                try:
                    with self._ocr_lock:
                        result = model.ocr(img_np, cls=True)
                except Exception as e2:
                    logger.error("PaddleOCR CPU fallback also failed: %s", e2)
                    return None
            else:
                logger.error("PaddleOCR failed: %s", e)
                return None

        # Debug: log raw result for troubleshooting
        logger.debug("PaddleOCR raw result type=%s, len=%s, snippet=%s",
                     type(result).__name__,
                     len(result) if isinstance(result, list) else 'N/A',
                     str(result)[:200] if result else 'None')

        # Basic validation
        if not result:
            return None

        # Handle None or empty result
        if result is None or (isinstance(result, list) and len(result) == 0):
            return None

        # PaddleOCR 3.x predict() returns dict with 'rec_texts', 'rec_scores', 'dt_polys'
        # PaddleOCR 2.x ocr() returns: [ [line1, line2, ...] ]
        page_result = None
        is_v3_format = False

        if isinstance(result, dict):
            # PaddleOCR 3.x format
            is_v3_format = True
            page_result = result
        elif isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], dict):
                # List of dicts (batch result from 3.x)
                is_v3_format = True
                page_result = result[0]
            else:
                # Legacy 2.x format
                page_result = result[0]
                if page_result is None:
                    return None
        else:
            page_result = result

        try:
            # Collect ALL lines with coordinates for robust sorting
            all_lines = []  # List of dicts

            # PaddleOCR 3.x format: dict with rec_texts, rec_scores, dt_polys
            if is_v3_format and isinstance(page_result, dict):
                texts = page_result.get('rec_texts', page_result.get('texts', []))
                scores = page_result.get('rec_scores', page_result.get('scores', []))
                polys = page_result.get('dt_polys', page_result.get('boxes', []))

                for idx, (text, score) in enumerate(zip(texts, scores)):
                    if not text:
                        continue
                    box = polys[idx] if idx < len(polys) else None
                    y_center = 0
                    x_left = 0
                    h_line = 20
                    if box is not None and len(box) >= 4:
                        try:
                            ys = [p[1] for p in box]
                            xs = [p[0] for p in box]
                            y_center = sum(ys) / len(ys)
                            x_left = min(xs)
                            h_line = max(ys) - min(ys)
                        except Exception:
                            pass
                    all_lines.append({
                        "y": y_center, "x": x_left, "h": h_line,
                        "text": str(text), "score": float(score), "box": box
                    })

            # PaddleOCR 2.x format: list of [box, (text, score)] tuples
            elif isinstance(page_result, list):
                for line in page_result:
                    if not isinstance(line, (list, tuple)) or len(line) < 2:
                        continue

                    # line = [ box, (text, confidence) ]
                    box = line[0]  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                    rec = line[1]  # ('text', confidence)

                    # Parse text and score
                    text = None
                    score = 0.0

                    if isinstance(rec, (list, tuple)) and len(rec) >= 2:
                        text = str(rec[0]) if rec[0] else ""
                        score = float(rec[1]) if rec[1] is not None else 0.0
                    elif isinstance(rec, str):
                        text = rec
                        score = 1.0

                    if not text:
                        continue

                    # Calculate Y position for sorting (2-line plates)
                    try:
                        if isinstance(box, (list, tuple)) and len(box) >= 4:
                            ys = [p[1] for p in box if isinstance(p, (list, tuple)) and len(p) >= 2]
                            xs = [p[0] for p in box if isinstance(p, (list, tuple)) and len(p) >= 2]
                            y_center = sum(ys) / len(ys) if ys else 0
                            x_left = min(xs) if xs else 0
                            h_line = (max(ys) - min(ys)) if ys else 20
                        else:
                            y_center = 0
                            x_left = 0
                            h_line = 20
                    except Exception:
                        y_center = 0
                        x_left = 0
                        h_line = 20

                    all_lines.append({
                        "y": y_center,
                        "x": x_left,
                        "h": h_line,
                        "text": text,
                        "score": score,
                        "box": box
                    })

            if not all_lines:
                return None

            # Sort by Y with tolerance (top-to-bottom), then X (left-to-right)
            all_lines.sort(key=lambda l: l["y"])

            lines_grouped = []
            current_line = [all_lines[0]]
            for i in range(1, len(all_lines)):
                prev = all_lines[i - 1]
                curr = all_lines[i]
                y_diff = abs(curr["y"] - prev["y"])
                h_avg = (prev["h"] + curr["h"]) / 2

                if y_diff < h_avg * 0.5:
                    current_line.append(curr)
                else:
                    current_line.sort(key=lambda l: l["x"])
                    lines_grouped.append(current_line)
                    current_line = [curr]

            current_line.sort(key=lambda l: l["x"])
            lines_grouped.append(current_line)

            # Merge all lines into single plate text
            merged_text = ""
            best_score = float('inf')  # Use MIN across lines — weakest line = true confidence
            worst_line_score = float('inf')
            combined_box = []

            for line_group in lines_grouped:
                for item in line_group:
                    clean_text = str(item["text"]).replace(" ", "").replace("-", "").replace(".", "").upper()
                    merged_text += clean_text
                    # Track worst line score — the weakest line limits overall confidence
                    if item["score"] < worst_line_score:
                        worst_line_score = item["score"]
                    if item["box"]:
                        combined_box = item["box"]

            best_score = worst_line_score if worst_line_score != float('inf') else 0.0

            if not merged_text or len(merged_text) < 4:
                return None

            return (merged_text, best_score, combined_box)

        except Exception as e:
            logger.error("Failed to parse OCR result: %s", e)
            return None

    def _apply_voting(self, track_id: int, text: str, score: float, area: int = 0) -> Optional[Tuple[str, float]]:
        """
        Add result to buffer and return the current weighted winner.
        Uses score + area weighting for best-shot selection.

        Returns:
            (plate_text, best_confidence) of the winning plate, or None if
            not enough votes yet.
        """
        now = time.time()

        with self._vote_lock:
            # Periodic cleanup of stagnant tracks (reliable interval-based)
            if now - self._last_cleanup_time > self.VOTE_CLEANUP_INTERVAL:
                self._last_cleanup_time = now
                stale_ids = [tid for tid, b in self._voting_buffer.items()
                            if not b or now - b[-1].get("ts", 0) > self.VOTE_BUFFER_TTL]
                for tid in stale_ids:
                    del self._voting_buffer[tid]

            if track_id not in self._voting_buffer:
                self._voting_buffer[track_id] = []

            buffer = self._voting_buffer[track_id]
            buffer.append({
                "text": text,
                "score": score,
                "area": area,
                "ts": now
            })

            # Keep buffer size manageable
            if len(buffer) > self.MAX_VOTE_HISTORY:
                buffer.pop(0)

            # Remove old entries from THIS buffer (TTL: 5 seconds)
            buffer[:] = [b for b in buffer if now - b["ts"] < self.VOTE_ENTRY_TTL]
            records = list(buffer)

        if not records:
            return None

        # DELAYED COMMIT LOGIC:
        # Require minimum readings before returning result
        if len(records) < self.MIN_VOTE_COUNT:
            return None

        # Check if we have any decent confidence reading
        max_score = max(e["score"] for e in records)
        if max_score < 0.65:
            # Allow if buffer has enough readings
            if len(records) < 3:
                return None

        # Calculate Weighted Votes using score² and area bonus
        vote_tally = Counter()
        raw_best = {}
        max_area = max(e["area"] for e in records) if records else 1

        for entry in records:
            # Weight = Score² * (1 + area_ratio)
            area_ratio = (entry["area"] / max_area) if max_area > 0 else 1
            weight = (entry["score"] ** 2) * (1 + area_ratio * 0.5)
            normalized = normalize_plate_basic(entry["text"])
            # Allow non-standard formats with reduced weight (don't discard entirely)
            if not is_valid_vn_plate_format(normalized):
                weight *= 0.3  # Penalize but don't exclude
            vote_tally[normalized] += weight
            if normalized not in raw_best or entry["score"] > raw_best[normalized][1]:
                raw_best[normalized] = (entry["text"], entry["score"])

        # Get winner
        if not vote_tally:
            return None
        winner_norm, _total_weight = vote_tally.most_common(1)[0]

        winner_text, winner_score = raw_best[winner_norm]
        return winner_text, winner_score

    def finalize_track(self, track_id: int) -> Optional[str]:
        """
        Get final consensus plate when vehicle exits frame.
        Sorts by (score, area) to select best readings.
        """
        with self._vote_lock:
            if track_id not in self._voting_buffer:
                return None

            records = self._voting_buffer[track_id]

            # Require minimum readings
            if len(records) < 3:
                return None

            # Sort by score DESC, then area DESC (best quality first)
            records.sort(key=lambda r: (r["score"], r["area"]), reverse=True)

            # Take top 5 best readings and vote (valid formats only)
            top_records = records[:5]
            texts = [normalize_plate_basic(r["text"]) for r in top_records
                     if is_valid_vn_plate_format(normalize_plate_basic(r["text"]))]

            if not texts:
                return None

            # Return most common among top readings
            winner_norm, _count = Counter(texts).most_common(1)[0]
            for r in top_records:
                normalized = normalize_plate_basic(r["text"])
                if normalized == winner_norm:
                    return r["text"]
            return None

    def clear_track(self, track_id: int):
        """Clear voting buffer when vehicle leaves frame"""
        with self._vote_lock:
            if track_id in self._voting_buffer:
                del self._voting_buffer[track_id]
