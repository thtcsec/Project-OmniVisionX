"""SimpleByteTrack - lightweight IoU-based tracker."""

from typing import Dict, List, Tuple

import numpy as np


def _iou(box_a: List[int], box_b: List[int]) -> float:
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    inter = max(0, x_b - x_a) * max(0, y_b - y_a)
    if inter == 0:
        return 0.0

    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


class SimpleByteTrack:
    """Greedy IoU tracker for demo use."""

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks: Dict[int, Dict[str, object]] = {}
        self.next_id = 1

    def update(self, detections: List[dict], frame_size: Tuple[int, int]) -> List[dict]:
        _ = frame_size
        if not detections:
            stale = []
            for tid, trk in self.tracks.items():
                trk["age"] = int(trk.get("age", 0)) + 1
                if trk["age"] > self.max_age:
                    stale.append(tid)
            for tid in stale:
                del self.tracks[tid]
            return []

        track_ids = list(self.tracks.keys())
        matched_det = set()
        matched_trk = set()
        results: List[dict] = []

        if track_ids:
            iou_matrix = np.zeros((len(detections), len(track_ids)), dtype=np.float32)
            for di, det in enumerate(detections):
                for ti, tid in enumerate(track_ids):
                    iou_matrix[di, ti] = _iou(det["bbox"], self.tracks[tid]["bbox"])

            for _ in range(min(len(detections), len(track_ids))):
                best = np.unravel_index(int(np.argmax(iou_matrix)), iou_matrix.shape)
                if iou_matrix[best] < self.iou_threshold:
                    break
                di, ti = int(best[0]), int(best[1])
                tid = track_ids[ti]
                det = detections[di].copy()
                det["track_id"] = tid
                results.append(det)

                self.tracks[tid]["bbox"] = det["bbox"]
                self.tracks[tid]["label"] = det["label"]
                self.tracks[tid]["age"] = 0

                matched_det.add(di)
                matched_trk.add(ti)
                iou_matrix[di, :] = -1.0
                iou_matrix[:, ti] = -1.0

        for di, det in enumerate(detections):
            if di in matched_det:
                continue
            tid = self.next_id
            self.next_id += 1
            det_copy = det.copy()
            det_copy["track_id"] = tid
            results.append(det_copy)
            self.tracks[tid] = {"bbox": det["bbox"], "label": det["label"], "age": 0}

        for ti, tid in enumerate(track_ids):
            if ti in matched_trk or tid not in self.tracks:
                continue
            self.tracks[tid]["age"] = int(self.tracks[tid].get("age", 0)) + 1
            if self.tracks[tid]["age"] > self.max_age:
                del self.tracks[tid]

        return results
