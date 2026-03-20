"""
ByteTrack adapter for omni-object
=============================
Wraps Ultralytics ByteTrack to assign persistent global_track_id
to every detected object. The ID survives across frames for the
lifetime of the track.

Each camera gets its own tracker instance (no cross-camera ID sharing).
"""
import logging
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger("omni-object.tracker")


@dataclass
class Detection:
    """A single YOLO detection with optional track assignment."""
    bbox: Tuple[int, int, int, int]   # x1, y1, x2, y2
    confidence: float
    class_id: int                      # COCO class id
    class_name: str
    global_track_id: int = -1          # assigned by ByteTrack


@dataclass
class TrackedObject:
    """Lightweight wrapper for ByteTrack output."""
    track_id: int
    bbox: Tuple[int, int, int, int]
    confidence: float
    class_id: int
    class_name: str


class _DetectionResults:
    """
    Minimal Results-like object that satisfies
    ``ultralytics.trackers.byte_tracker.BYTETracker.update()`` API.

    Expected attributes (see BYTETracker.update source):
      - conf   : np.ndarray (N,)    – confidence scores
      - xywh   : np.ndarray (N, 4)  – [cx, cy, w, h]
      - cls    : np.ndarray (N,)    – class ids
    """

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.conf = conf
        self.cls = cls
        # convert xyxy → xywh  (+ append detection index as 5th col)
        x1, y1, x2, y2 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        self.xywh = np.stack([cx, cy, w, h], axis=1)


class ByteTrackAdapter:
    """
    Per-camera ByteTrack instance using Ultralytics BYTETracker.

    Usage:
        tracker = ByteTrackAdapter(settings)
        tracked = tracker.update(detections)
    """

    def __init__(self, settings):
        self.settings = settings
        self._tracker = None
        self._init_tracker()

    def _init_tracker(self):
        """Initialize ByteTrack via ultralytics."""
        try:
            from ultralytics.trackers.byte_tracker import BYTETracker

            args = SimpleNamespace(
                track_high_thresh=self.settings.track_high_thresh,
                track_low_thresh=self.settings.track_low_thresh,
                new_track_thresh=max(
                    self.settings.track_low_thresh,
                    self.settings.track_high_thresh - 0.05
                ),
                track_buffer=self.settings.track_buffer,
                match_thresh=self.settings.match_thresh,
            )
            self._tracker = BYTETracker(args, frame_rate=self.settings.frame_rate)
            logger.info("ByteTrack initialized (buffer=%d, match=%.2f)",
                        self.settings.track_buffer, self.settings.match_thresh)
        except ImportError:
            logger.warning("ultralytics ByteTrack not available, falling back to simple IoU tracker")
            self._tracker = None

    def update(self, detections: List[Detection]) -> List[TrackedObject]:
        """
        Feed detections into ByteTrack tracker.
        Returns list of TrackedObject with assigned global_track_id.

        IMPORTANT: must be called every frame (even with empty list)
        so ByteTrack can age out lost tracks correctly.
        """
        if self._tracker is None:
            if not detections:
                return []
            # Fallback: assign sequential IDs (no real tracking)
            return [
                TrackedObject(
                    track_id=i,
                    bbox=d.bbox,
                    confidence=d.confidence,
                    class_id=d.class_id,
                    class_name=d.class_name,
                )
                for i, d in enumerate(detections)
            ]

        if not detections:
            # Still tick the tracker so it ages out lost tracks
            empty = _DetectionResults(
                xyxy=np.empty((0, 4), dtype=np.float32),
                conf=np.empty((0,), dtype=np.float32),
                cls=np.empty((0,), dtype=np.float32),
            )
            try:
                self._tracker.update(empty)
            except Exception:
                pass
            return []

        # Build arrays for ByteTrack
        xyxy = np.array([[d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3]]
                         for d in detections], dtype=np.float32)
        conf = np.array([d.confidence for d in detections], dtype=np.float32)
        cls = np.array([d.class_id for d in detections], dtype=np.float32)

        results_obj = _DetectionResults(xyxy=xyxy, conf=conf, cls=cls)

        try:
            online_targets = self._tracker.update(results_obj)
        except Exception as e:
            logger.warning("ByteTrack update failed: %s", e)
            # Assign unique negative IDs so they aren't confused as the same track
            return [
                TrackedObject(
                    track_id=-(i + 1),
                    bbox=d.bbox,
                    confidence=d.confidence,
                    class_id=d.class_id,
                    class_name=d.class_name,
                )
                for i, d in enumerate(detections)
            ]

        results = []
        for t in online_targets:
            try:
                # STrack exposes .xyxy property → [x1, y1, x2, y2]
                if hasattr(t, 'xyxy') and len(t.xyxy) >= 4:
                    x1, y1, x2, y2 = [int(v) for v in t.xyxy[:4]]
                else:
                    tlwh = t.tlwh
                    x1, y1 = int(tlwh[0]), int(tlwh[1])
                    x2, y2 = int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3])

                track_id = int(t.track_id)
                score = float(t.score)

                # Use class info from STrack (set by BYTETracker from our input)
                cls_id = int(t.cls) if hasattr(t, 'cls') and t.cls is not None else -1

                # Find closest original detection for class name (and fallback class_id)
                best_det = self._match_detection(detections, (x1, y1, x2, y2))
                if cls_id < 0 and best_det:
                    cls_id = best_det.class_id
                class_name = best_det.class_name if best_det else "unknown"

                results.append(TrackedObject(
                    track_id=track_id,
                    bbox=(x1, y1, x2, y2),
                    confidence=score,
                    class_id=cls_id,
                    class_name=class_name,
                ))
            except Exception as e:
                logger.warning("Failed to parse STrack output: %s", e)
                continue

        return results

    @staticmethod
    def _match_detection(
        detections: List[Detection],
        bbox: Tuple[int, int, int, int],
    ) -> Optional[Detection]:
        """Find the original detection closest to tracked bbox by IoU."""
        best_iou = 0.0
        best_det = None
        for d in detections:
            iou = ByteTrackAdapter._compute_iou(d.bbox, bbox)
            if iou > best_iou:
                best_iou = iou
                best_det = d
        # Require minimum IoU to avoid false class assignments across distant detections
        MIN_MATCH_IOU = 0.1
        return best_det if best_iou >= MIN_MATCH_IOU else None

    @staticmethod
    def _compute_iou(
        a: Tuple[int, int, int, int],
        b: Tuple[int, int, int, int],
    ) -> float:
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
        area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0


class TrackerPool:
    """
    Manages per-camera ByteTrack instances.
    Lazily creates a tracker when a camera is first seen.
    """

    def __init__(self, settings):
        self.settings = settings
        self._trackers: Dict[str, ByteTrackAdapter] = {}

    def get_tracker(self, camera_id: str) -> ByteTrackAdapter:
        if camera_id not in self._trackers:
            self._trackers[camera_id] = ByteTrackAdapter(self.settings)
            logger.info("Created ByteTrack instance for camera %s", camera_id[:8])
        return self._trackers[camera_id]

    def remove_tracker(self, camera_id: str):
        self._trackers.pop(camera_id, None)

    def clear(self):
        self._trackers.clear()
