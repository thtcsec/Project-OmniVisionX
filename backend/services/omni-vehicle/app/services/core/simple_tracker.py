"""
Simple IOU-based tracker for plate detections.
Keeps consistent track IDs across frames for temporal voting.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


BBox = Tuple[int, int, int, int]
Detection = Tuple[BBox, float]


@dataclass
class _Track:
    track_id: int
    bbox: BBox
    last_seen: float
    age: int = 0
    hits: int = 1
    confirmed: bool = False


class SimpleTracker:
    def __init__(self, iou_threshold: float = 0.3, max_age_seconds: float = 1.0, min_hits: int = 2) -> None:
        self.iou_threshold = iou_threshold
        self.max_age_seconds = max_age_seconds
        self.min_hits = min_hits
        self._next_id = 1
        self._tracks: Dict[int, _Track] = {}

    def update(self, detections: Iterable[BBox | Detection], now: Optional[float] = None,
               include_unconfirmed: bool = False) -> List[Tuple[BBox, int]]:
        """Assign track IDs to detections using global greedy IOU matching."""
        if now is None:
            now = time.time()

        det_list, scores = _normalize_detections(detections)

        # Age existing tracks
        for track in self._tracks.values():
            track.age += 1

        assignments: List[Tuple[BBox, int]] = []

        # Build all candidate pairs above IOU threshold
        pairs: List[Tuple[float, float, int, BBox]] = []
        for track_id, track in self._tracks.items():
            for det, score in zip(det_list, scores):
                iou = _bbox_iou(track.bbox, det)
                if iou >= self.iou_threshold:
                    pairs.append((iou, score, track_id, det))

        # Sort by IOU then score
        pairs.sort(key=lambda x: (x[0], x[1]), reverse=True)

        used_tracks = set()
        used_dets = set()

        for iou, score, track_id, det in pairs:
            if track_id in used_tracks or det in used_dets:
                continue
            track = self._tracks[track_id]
            track.bbox = det
            track.last_seen = now
            track.age = 0
            track.hits += 1
            if track.hits >= self.min_hits:
                track.confirmed = True
            if track.confirmed or include_unconfirmed:
                assignments.append((det, track_id))
            used_tracks.add(track_id)
            used_dets.add(det)

        # Cleanup stale tracks by time
        for track_id, track in list(self._tracks.items()):
            if now - track.last_seen > self.max_age_seconds:
                self._tracks.pop(track_id, None)

        # Create tracks for unmatched detections
        for det, score in zip(det_list, scores):
            if det in used_dets:
                continue
            track_id = self._next_id
            self._next_id += 1
            self._tracks[track_id] = _Track(
                track_id=track_id,
                bbox=det,
                last_seen=now,
                age=0,
                hits=1,
                confirmed=self.min_hits <= 1,
            )
            if self._tracks[track_id].confirmed or include_unconfirmed:
                assignments.append((det, track_id))

        return assignments

    def clear(self) -> None:
        self._tracks.clear()
        self._next_id = 1


class TrackerManager:
    """Maintain trackers per camera."""

    def __init__(self) -> None:
        self._trackers: Dict[str, SimpleTracker] = {}
        self._last_access: Dict[str, float] = {}
        self._last_cleanup: float = 0.0

    def get_tracker(
        self,
        camera_id: str,
        iou_threshold: float = 0.3,
        max_age_seconds: float = 1.0,
        min_hits: int = 2,
    ) -> SimpleTracker:
        now = time.time()
        self._last_access[camera_id] = now
        
        # Time-based cleanup instead of modulo (which skips many counts)
        if now - self._last_cleanup > 300:  # Every 5 minutes
            self._last_cleanup = now
            stale = [cid for cid, ts in self._last_access.items() if now - ts > 3600]
            for cid in stale:
                self._trackers.pop(cid, None)
                self._last_access.pop(cid, None)

        if camera_id not in self._trackers:
            self._trackers[camera_id] = SimpleTracker(
                iou_threshold=iou_threshold,
                max_age_seconds=max_age_seconds,
                min_hits=min_hits,
            )
        return self._trackers[camera_id]

    def clear(self, camera_id: Optional[str] = None) -> None:
        if camera_id is None:
            self._trackers.clear()
            self._last_access.clear()
        else:
            self._trackers.pop(camera_id, None)
            self._last_access.pop(camera_id, None)


def _bbox_iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0
    return inter_area / union


def _normalize_detections(detections: Iterable[BBox | Detection]) -> Tuple[List[BBox], List[float]]:
    det_list: List[BBox] = []
    scores: List[float] = []
    for det in detections:
        if isinstance(det, tuple) and len(det) == 2 and isinstance(det[0], tuple):
            bbox, score = det
            det_list.append(bbox)
            scores.append(float(score))
        else:
            det_list.append(det)  # type: ignore[arg-type]
            scores.append(0.0)
    return det_list, scores
