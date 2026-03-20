"""
Tracking consensus + deduplication logic for omni-vehicle.
Migrated from root ai-engine/app/workers/tracking_consensus.py
"""
import asyncio
import logging
import os
import time
from collections import Counter
from typing import Optional, Tuple, Dict

from app.services.plate.plate_utils import normalize_vn_plate_confusions, plate_edit_distance

logger = logging.getLogger(__name__)


class TrackingConsensus:
    """Track-based and temporal deduplication for plates."""

    def __init__(self, settings):
        self.settings = settings

        # Plate deduplication cache: {"camera_plate": (timestamp, best_conf, bbox_center)}
        self._plate_cache: Dict[str, tuple] = {}
        self._plate_cache_lock = asyncio.Lock()

        # Track-based deduplication cache (Primary)
        self._track_cache: Dict[str, dict] = {}
        self._track_cache_lock = asyncio.Lock()

        # Deterministic cleanup counter
        self._cleanup_counter = 0
        self._cleanup_interval = 10

        self._cache_ttl = float(os.getenv("PLATE_CACHE_TTL", "15"))
        self._parked_ttl = float(os.getenv("PARKED_DEDUP_TTL", "600"))
        self._conf_epsilon = 0.05
        self._position_threshold = float(os.getenv("PLATE_POSITION_THRESHOLD", "250"))
        self._stationary_threshold = float(os.getenv("PLATE_STATIONARY_THRESHOLD", "140"))
        self._position_vote_min_ratio = 0.60
        self._max_missing_positions = 1
        self._jump_dampen = float(os.getenv("PLATE_CONF_JUMP_DAMPEN", "0.9"))
        self._high_confidence_threshold = float(os.getenv("PLATE_HIGH_CONF_THRESHOLD", "0.85"))
        self._confirm_conf = float(os.getenv("PLATE_CONFIRM_CONF", "0.8"))
        self._max_state_misses = int(os.getenv("PLATE_CONFIRM_MAX_MISSES", "2"))
        # Timeout (seconds) for DEFERRED tracks: if a plate stays DEFERRED
        # longer than this AND has at least 1 hit, emit the best result anyway.
        # Prevents stalled vehicles (traffic lights, parking) from being missed.
        self._deferred_timeout = float(os.getenv("PLATE_DEFERRED_TIMEOUT", "2.5"))
        self._track_stale_reset_s = float(os.getenv("PLATE_TRACK_STALE_RESET_S", "3.0"))

    def _runtime_window_size(self) -> int:
        return max(3, int(os.getenv("PLATE_CONSENSUS_WINDOW", "5")))

    def _runtime_confirm_hits(self) -> int:
        return max(
            1,
            int(os.getenv("PLATE_CONFIRM_HITS", str(getattr(self.settings, "event_min_vote_count", 1))))
        )

    def _runtime_max_history(self) -> int:
        return max(
            3,
            int(os.getenv("PLATE_CONSENSUS_HISTORY", str(getattr(self.settings, "lpr_consensus_history", 12))))
        )

    def _runtime_relaxed_instant_margin(self) -> float:
        return max(0.0, min(0.2, float(os.getenv("PLATE_INSTANT_RELAX_MARGIN", "0.10"))))

    def _runtime_relaxed_instant_sharpness(self) -> float:
        return max(0.0, float(os.getenv("PLATE_INSTANT_RELAX_SHARPNESS", "110")))

    @staticmethod
    def normalize_plate(plate: str) -> str:
        return normalize_vn_plate_confusions(plate)

    async def cleanup_plate_cache(self):
        """Remove expired entries from caches."""
        self._cleanup_counter += 1
        if self._cleanup_counter < self._cleanup_interval:
            return
        self._cleanup_counter = 0

        now = time.time()
        expired_plate_count = 0
        expired_track_count = 0

        try:
            async with self._plate_cache_lock:
                expired_plate = [k for k, v in self._plate_cache.items() if (now - v[0]) > self._cache_ttl]
                for k in expired_plate:
                    del self._plate_cache[k]
                expired_plate_count = len(expired_plate)

            async with self._track_cache_lock:
                expired_track = [k for k, v in self._track_cache.items() if (now - v["timestamp"]) > self._cache_ttl]
                for k in expired_track:
                    entry = self._track_cache[k]
                    logger.info("🏁 FINAL PLATE [%s]: %s (Conf: %.2f)", k, entry['plate'], entry['confidence'])
                    del self._track_cache[k]
                expired_track_count = len(expired_track)

                # Enforce max cache size — evict oldest entries if cache too large
                max_cache = self._runtime_max_history() * 100  # e.g. 10 * 100 = 1000 tracks
                if len(self._track_cache) > max_cache:
                    sorted_keys = sorted(
                        self._track_cache.keys(),
                        key=lambda k: self._track_cache[k].get("timestamp", 0)
                    )
                    to_evict = sorted_keys[:len(self._track_cache) - max_cache]
                    for k in to_evict:
                        del self._track_cache[k]
                    if to_evict:
                        logger.info("🧹 Evicted %d stale tracks (cache overflow)", len(to_evict))
        except Exception:
            logger.exception("Failed to cleanup consensus caches")

        if expired_plate_count or expired_track_count:
            logger.info("🧹 Cleaned cache: %d plates, %d tracks", expired_plate_count, expired_track_count)

    async def should_process_plate(
        self,
        camera_id: str,
        plate_text: str,
        confidence: float,
        thumbnail_path: str,
        full_frame_path: str,
        bbox: list = None,
        track_id: int = -1,
        sharpness: float = 0.0,
    ) -> Tuple[bool, Optional[dict]]:
        """
        Decide whether to PROCESS (Insert/Update) or SKIP a detected plate.
        Uses temporal consensus logic for tracked vehicles.
        Returns: (should_process, cache_entry)
        """
        now = time.time()

        bbox_center = (0, 0)
        if bbox and len(bbox) >= 4:
            bbox_center = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))

        async def _stationary_emit_allowed(norm_plate: str) -> bool:
            key = f"{camera_id}_{norm_plate}"
            async with self._plate_cache_lock:
                if key in self._plate_cache:
                    last_ts, _last_conf, last_center = self._plate_cache[key]
                    dist = ((bbox_center[0] - last_center[0])**2 + (bbox_center[1] - last_center[1])**2) ** 0.5
                    if dist <= self._stationary_threshold and (now - last_ts) < self._parked_ttl:
                        self._plate_cache[key] = (now, confidence, bbox_center)
                        return False
                self._plate_cache[key] = (now, confidence, bbox_center)
                return True

        # === 1. TRACK-BASED DEDUPLICATION (PRIMARY) ===
        if track_id is not None and track_id != -1:
            track_key = f"{camera_id}_{track_id}"

            async with self._track_cache_lock:
                entry = self._track_cache.get(track_key)
                plate_norm = self.normalize_plate(plate_text)

                if entry:
                    entry_age = now - float(entry.get("timestamp", now))
                    if entry_age > self._track_stale_reset_s:
                        logger.debug(
                            "♻️ Reset stale track state: %s age=%.2fs > %.2fs",
                            track_key, entry_age, self._track_stale_reset_s,
                        )
                        self._track_cache.pop(track_key, None)
                        entry = None

                if entry:
                    prev_plate_norm = self.normalize_plate(entry.get("plate", ""))
                    if prev_plate_norm and plate_edit_distance(prev_plate_norm, plate_norm) == 1:
                        if entry.get("confidence", 0.0) >= self._high_confidence_threshold:
                            confidence = max(0.0, confidence * self._jump_dampen)
                    history = entry.get("history", [])
                    counts = entry.get("counts", {})
                    conf_sums = entry.get("conf_sums", {})
                    history.append({
                        "plate": plate_norm,
                        "raw_plate": plate_text,
                        "confidence": confidence,
                        "timestamp": now,
                        "sharpness": sharpness,
                    })

                    # Best-shot tracking: only update thumbnail when frame is sharper
                    if sharpness > entry.get("best_sharpness", 0.0):
                        entry["best_sharpness"] = sharpness
                        entry["thumb_path"] = thumbnail_path
                        entry["full_path"] = full_frame_path
                        logger.debug("📸 Best-shot update: track=%s sharpness=%.0f", track_key, sharpness)
                    counts[plate_norm] = counts.get(plate_norm, 0) + 1
                    conf_sums[plate_norm] = conf_sums.get(plate_norm, 0.0) + confidence
                    if len(history) > self._runtime_window_size():
                        old = history.pop(0)
                        old_plate = old.get("plate")
                        old_conf = old.get("confidence", 0.0)
                        if old_plate:
                            counts[old_plate] = counts.get(old_plate, 0) - 1
                            if counts.get(old_plate, 0) <= 0:
                                counts.pop(old_plate, None)
                                conf_sums.pop(old_plate, None)
                            else:
                                conf_sums[old_plate] = conf_sums.get(old_plate, 0.0) - old_conf
                    entry["history"] = history
                    entry["timestamp"] = now
                    entry["counts"] = counts
                    entry["conf_sums"] = conf_sums

                    merged_counts = {}
                    merged_conf_sums = {}
                    for p, count in counts.items():
                        norm_key = self._normalize_plate_for_consensus(p, history)
                        merged_counts[norm_key] = merged_counts.get(norm_key, 0) + count
                        merged_conf_sums[norm_key] = merged_conf_sums.get(norm_key, 0.0) + conf_sums.get(p, 0.0)

                    best_norm = self.normalize_plate(entry["plate"])
                    best_combined_score = -1.0

                    for p in merged_counts:
                        score = merged_conf_sums[p]
                        if score > best_combined_score:
                            best_combined_score = score
                            best_norm = p

                    voted_plate, is_complete, missing_positions = self._vote_by_position(history)
                    allow_relaxed_vote = (
                        voted_plate
                        and len(voted_plate) >= 7
                        and missing_positions <= (self._max_missing_positions + 1)
                    )
                    best_plate = voted_plate if (voted_plate and (is_complete or allow_relaxed_vote)) else best_norm
                    avg_conf = (merged_conf_sums.get(best_norm, 0.0) / merged_counts[best_norm]) if merged_counts.get(best_norm) else confidence

                    self._update_track_state(entry, best_norm, avg_conf, is_complete)

                    # Use same merge normalization on prev plate for consistent comparison
                    prev_plate_norm = self._normalize_plate_for_consensus(
                        self.normalize_plate(entry["plate"]), history
                    )

                    if entry.get("event_id") == "DEFERRED":
                        if entry.get("state") == "CONFIRMED":
                            winner_count = merged_counts.get(best_norm, 0)
                            if not await _stationary_emit_allowed(best_norm):
                                return False, None
                            logger.info("✅ Consensus reached: %s | %s (votes=%d)", track_key, best_plate, winner_count)
                            entry["plate"] = best_plate
                            entry["confidence"] = avg_conf
                            entry["best_score"] = best_combined_score
                            entry["initial_confidence"] = avg_conf
                            entry["event_id"] = "PENDING"
                            entry["plate_id"] = "PENDING"
                            # thumb_path/full_path already set by best-shot tracking above
                            return True, entry

                        # Timeout fallback: emit best result if DEFERRED too long.
                        # Prevents stalled vehicles (traffic lights, slow-moving)
                        # from being silently dropped when they never accumulate
                        # enough hits for CONFIRMED state.
                        entry_age = now - entry.get("timestamp_created", entry.get("timestamp", now))
                        entry_hits = entry.get("hits", 0)
                        if entry_age >= self._deferred_timeout and entry_hits >= 1:
                            if not await _stationary_emit_allowed(best_norm):
                                return False, None
                            logger.info(
                                "⏰ Deferred timeout: %s | %s (age=%.1fs, hits=%d, conf=%.2f)",
                                track_key, best_plate, entry_age, entry_hits, avg_conf,
                            )
                            entry["plate"] = best_plate
                            entry["confidence"] = avg_conf
                            entry["best_score"] = best_combined_score
                            entry["initial_confidence"] = avg_conf
                            entry["event_id"] = "PENDING"
                            entry["plate_id"] = "PENDING"
                            # thumb_path/full_path already set by best-shot tracking
                            return True, entry

                        if confidence > entry["confidence"]:
                            entry["plate"] = plate_text
                            entry["confidence"] = confidence
                            # thumb_path/full_path managed by best-shot tracking
                        return False, None

                    if entry.get("event_id") == "PENDING" or entry.get("event_id") is None:
                        if confidence > entry["confidence"]:
                            entry["plate"] = plate_text
                            entry["confidence"] = confidence
                            # thumb_path/full_path managed by best-shot tracking
                            # Re-attempt DB write if previous write may have failed and confidence improved
                            if confidence - entry.get("initial_confidence", 0.0) > 0.15:
                                entry["initial_confidence"] = confidence
                                return True, entry
                        return False, None

                    last_db_update = entry.get("last_db_update", 0)
                    is_new_winner = best_norm != prev_plate_norm
                    improved_significantly = (
                        best_combined_score > (entry.get("best_score", 0) + 1.0) and
                        (now - last_db_update) > 2.0
                    )

                    if (is_new_winner or improved_significantly) and is_complete:
                        logger.info(
                            "📈 Consensus Update: %s | %s -> %s (Score: %.1f)",
                            track_key, entry['plate'], best_plate, best_combined_score
                        )

                        old_files = {
                            "event_id": entry["event_id"],
                            "plate_id": entry["plate_id"],
                            "old_thumb": entry.get("thumb_path"),
                            "old_full": entry.get("full_path")
                        }

                        entry["plate"] = best_plate
                        entry["confidence"] = merged_conf_sums[best_norm] / merged_counts[best_norm]
                        entry["best_score"] = best_combined_score
                        entry["last_db_update"] = now
                        # thumb_path/full_path managed by best-shot tracking

                        return True, old_files

                    return False, None

                # New Track -> INSERT
                min_instant = self.settings.event_instant_confidence
                relaxed_margin = self._runtime_relaxed_instant_margin()
                relaxed_sharpness = self._runtime_relaxed_instant_sharpness()
                allow_relaxed_instant = (
                    confidence >= max(0.0, min_instant - relaxed_margin)
                    and sharpness >= relaxed_sharpness
                )
                if confidence >= min_instant or allow_relaxed_instant:
                    if not await _stationary_emit_allowed(plate_norm):
                        self._track_cache[track_key] = {
                            "plate": plate_text,
                            "confidence": confidence,
                            "timestamp": now,
                            "timestamp_created": now,
                            "bbox_center": bbox_center,
                            "event_id": "DEFERRED",
                            "plate_id": "DEFERRED",
                            "thumb_path": thumbnail_path,
                            "full_path": full_frame_path,
                            "best_sharpness": sharpness,
                            "history": [{"plate": plate_norm, "raw_plate": plate_text, "confidence": confidence, "timestamp": now, "sharpness": sharpness}],
                            "counts": {plate_norm: 1},
                            "conf_sums": {plate_norm: confidence},
                            "best_score": confidence,
                            "last_db_update": 0,
                            "state": "TRACKING",
                            "hits": 1,
                            "misses": 0,
                            "last_plate_norm": plate_norm,
                            "hit_conf_sum": confidence
                        }
                        return False, None
                    self._track_cache[track_key] = {
                        "plate": plate_text,
                        "confidence": confidence,
                        "timestamp": now,
                        "bbox_center": bbox_center,
                        "event_id": "PENDING",
                        "plate_id": "PENDING",
                        "thumb_path": thumbnail_path,
                        "full_path": full_frame_path,
                        "best_sharpness": sharpness,
                        "history": [{"plate": plate_norm, "raw_plate": plate_text, "confidence": confidence, "timestamp": now, "sharpness": sharpness}],
                        "counts": {plate_norm: 1},
                        "conf_sums": {plate_norm: confidence},
                        "best_score": confidence,
                        "last_db_update": now,
                        "state": "CONFIRMED",
                        "hits": self._runtime_confirm_hits(),
                        "misses": 0,
                        "last_plate_norm": plate_norm,
                        "hit_conf_sum": confidence
                    }
                    if allow_relaxed_instant and confidence < min_instant:
                        logger.info(
                            "⚡ Relaxed instant emit: %s conf=%.2f (min=%.2f, sharpness=%.1f)",
                            track_key, confidence, min_instant, sharpness,
                        )
                    return True, self._track_cache[track_key]

                self._track_cache[track_key] = {
                    "plate": plate_text,
                    "confidence": confidence,
                    "timestamp": now,
                    "timestamp_created": now,
                    "bbox_center": bbox_center,
                    "event_id": "DEFERRED",
                    "plate_id": "DEFERRED",
                    "thumb_path": thumbnail_path,
                    "full_path": full_frame_path,
                    "best_sharpness": sharpness,
                    "history": [{"plate": plate_norm, "raw_plate": plate_text, "confidence": confidence, "timestamp": now, "sharpness": sharpness}],
                    "counts": {plate_norm: 1},
                    "conf_sums": {plate_norm: confidence},
                    "best_score": confidence,
                    "last_db_update": 0,
                    "state": "TRACKING",
                    "hits": 1,
                    "misses": 0,
                    "last_plate_norm": plate_norm,
                    "hit_conf_sum": confidence
                }
                logger.info("⏳ Deferred: %s conf=%.2f < %.2f (waiting for consensus)", track_key, confidence, min_instant)
                return False, None

        # === 2. FALLBACK: LEGACY SPATIAL/TEMPORAL DEDUP ===
        normalized = self.normalize_plate(plate_text)
        key = f"{camera_id}_{normalized}"

        async with self._plate_cache_lock:
            if key in self._plate_cache:
                last_ts, last_conf, last_center = self._plate_cache[key]
                dist = ((bbox_center[0] - last_center[0])**2 + (bbox_center[1] - last_center[1])**2) ** 0.5

                if dist <= self._stationary_threshold and (now - last_ts) < self._parked_ttl:
                    self._plate_cache[key] = (now, confidence, bbox_center)
                    return False, None

                if (now - last_ts) < self._cache_ttl:
                    if dist > self._position_threshold:
                        self._plate_cache[key] = (now, confidence, bbox_center)
                        return True, None
                    return False, None

            self._plate_cache[key] = (now, confidence, bbox_center)
            return True, None

    def _normalize_plate_for_consensus(self, plate_norm: str, history: list) -> str:
        """Map incomplete OCR variants to stronger consensus candidates.

        OCR frequently drops 1-2 digits in the middle of the serial block.
        We only up-map to a longer candidate when the shorter text is a very
        strong reduction variant of a longer one already seen for the same track.
        """
        if not plate_norm:
            return plate_norm

        norm = plate_norm
        if len(norm) < 8:
            candidates = [h.get("plate", "") for h in history]
            longer = [
                c for c in candidates
                if c
                and len(c) in (len(norm) + 1, len(norm) + 2)
                and (
                    c.startswith(norm)
                    or self._is_reduction_variant(norm, c)
                )
            ]
            if longer:
                best_candidate = Counter(longer).most_common(1)[0][0]
                return best_candidate
        return norm

    @staticmethod
    def _is_reduction_variant(shorter: str, longer: str) -> bool:
        """Return True when `shorter` looks like a digit-drop OCR variant.

        Requirements are intentionally strict to avoid merging unrelated plates:
        - longer is only 1-2 chars longer
        - same province + series prefix (first 5 chars when available)
        - same 2-char suffix when available
        - shorter is a subsequence of longer
        """
        if not shorter or not longer or len(longer) <= len(shorter):
            return False

        length_gap = len(longer) - len(shorter)
        if length_gap < 1 or length_gap > 2:
            return False

        if len(shorter) >= 5 and shorter[:5] != longer[:5]:
            return False

        if len(shorter) >= 2 and shorter[-2:] != longer[-2:]:
            return False

        index = 0
        for ch in longer:
            if index < len(shorter) and shorter[index] == ch:
                index += 1
        return index == len(shorter)

    # _edit_distance → moved to plate_utils.plate_edit_distance

    def _vote_by_position(self, history: list) -> Tuple[str, bool, int]:
        """Vote per character position to mitigate partial occlusion."""
        normalized_history = []
        for item in history:
            plate = item.get("plate", "")
            if not plate:
                continue
            normalized_history.append({
                **item,
                "plate": self._normalize_plate_for_consensus(plate, history),
            })

        texts = [h.get("plate", "") for h in normalized_history if h.get("plate")]
        if not texts:
            return "", False, 0

        length_weights: Dict[int, float] = {}
        for item in normalized_history:
            plate = item.get("plate", "")
            if not plate:
                continue
            weight = float(item.get("confidence", 0.0))
            length_weights[len(plate)] = length_weights.get(len(plate), 0.0) + weight

        dominant_len = max(length_weights.items(), key=lambda pair: (pair[1], pair[0]))[0]
        vote_history = [h for h in normalized_history if len(h.get("plate", "")) == dominant_len]
        if not vote_history:
            return "", False, 0

        max_len = dominant_len
        if max_len == 0:
            return "", False, 0

        position_votes = [dict() for _ in range(max_len)]
        position_counts = [0.0 for _ in range(max_len)]

        for h in vote_history:
            t = h.get("plate", "")
            conf = float(h.get("confidence", 0.0))
            # Sharpness-weighted voting: sharp frames dominate blurry ones
            sharp = float(h.get("sharpness", 0.0))
            sharp_factor = 0.3 + 0.7 * min(sharp / 200.0, 1.0)
            weight = conf * sharp_factor
            for i in range(max_len):
                if i < len(t):
                    position_counts[i] += weight
                    ch = t[i]
                    position_votes[i][ch] = position_votes[i].get(ch, 0.0) + weight

        result_chars = []
        missing_positions = 0

        for i in range(max_len):
            if not position_votes[i]:
                missing_positions += 1
                result_chars.append("?")
                continue

            best_char, best_count = max(position_votes[i].items(), key=lambda kv: kv[1])
            coverage = best_count / max(1e-6, position_counts[i])
            if coverage < self._position_vote_min_ratio:
                missing_positions += 1
                # Still emit the best available char (low confidence) instead of '?',
                # so callers get a usable string when _max_missing_positions allows misses.
                result_chars.append(best_char)
            else:
                result_chars.append(best_char)

        voted = "".join(result_chars)
        # Previously also required "?" not in voted, which made _max_missing_positions
        # useless (missing slots always appended '?' → condition was always False for any miss).
        # Fixed: check missing_positions count only; emit best char for low-coverage slots.
        is_complete = (
            missing_positions <= self._max_missing_positions and
            len(voted) >= 7  # Match ocr_min_text_length; was 8, blocking valid 7-char plates
        )
        return voted, is_complete, missing_positions

    def _update_track_state(self, entry: dict, best_norm: str, avg_conf: float, is_complete: bool) -> str:
        last_norm = entry.get("last_plate_norm") or best_norm
        if best_norm == last_norm:
            entry["hits"] = entry.get("hits", 0) + 1
            entry["hit_conf_sum"] = entry.get("hit_conf_sum", 0.0) + avg_conf
            entry["misses"] = 0
        else:
            entry["misses"] = entry.get("misses", 0) + 1
            if entry["misses"] >= self._max_state_misses:
                entry["last_plate_norm"] = best_norm
                entry["hits"] = 1
                entry["hit_conf_sum"] = avg_conf
                entry["misses"] = 0

        if is_complete and avg_conf >= self._confirm_conf:
            if entry.get("hits", 0) >= self._runtime_confirm_hits():
                entry["state"] = "CONFIRMED"
            else:
                entry["state"] = "CONFIRMING"
        else:
            current_state = entry.get("state", "TRACKING")
            misses = entry.get("misses", 0)
            # Allow revert from CONFIRMED if too many consecutive misses (plate changed/occluded)
            if current_state == "CONFIRMED" and misses >= self._max_state_misses * 2:
                entry["state"] = "TRACKING"
                entry["hits"] = 0
            elif current_state != "CONFIRMED":
                entry["state"] = "TRACKING"

        return entry.get("state", "TRACKING")

    def get_stats(self) -> dict:
        return {
            "track_cache_size": len(self._track_cache),
            "plate_cache_size": len(self._plate_cache),
        }
