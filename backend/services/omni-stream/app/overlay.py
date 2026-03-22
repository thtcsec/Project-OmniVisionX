"""Server-side bbox drawing."""

from typing import List

import cv2
import numpy as np

COLORS = {
    "car": (255, 229, 0),
    "truck": (0, 145, 255),
    "bus": (0, 234, 255),
    "motorcycle": (251, 64, 224),
    "person": (3, 255, 118),
    "bicycle": (82, 82, 255),
}
DEFAULT_COLOR = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_detections(frame: np.ndarray, detections: List[dict]) -> np.ndarray:
    overlay = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.get("bbox", [0, 0, 0, 0])]
        label = str(det.get("label", "?"))
        conf = float(det.get("confidence", 0.0))
        track_id = det.get("track_id")
        plate = det.get("plate_number")

        color = COLORS.get(label.lower(), DEFAULT_COLOR)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        parts = [label]
        if conf > 0:
            parts.append(f"{conf * 100:.0f}%")
        if track_id is not None:
            parts.append(f"#{track_id}")
        label_text = " ".join(parts)

        (tw, th), _ = cv2.getTextSize(label_text, FONT, 0.55, 1)
        label_y = max(y1 - 8, th + 4)
        cv2.rectangle(overlay, (x1, label_y - th - 4), (x1 + tw + 8, label_y + 4), color, -1)
        cv2.putText(overlay, label_text, (x1 + 4, label_y), FONT, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        if plate:
            plate_text = f"PLATE: {plate}"
            (pw, ph), _ = cv2.getTextSize(plate_text, FONT, 0.6, 2)
            py = min(y2 + ph + 10, frame.shape[0] - 5)
            cv2.rectangle(overlay, (x1, py - ph - 4), (x1 + pw + 8, py + 4), (0, 0, 0), -1)
            cv2.putText(overlay, plate_text, (x1 + 4, py), FONT, 0.6, (0, 234, 255), 2, cv2.LINE_AA)

    return cv2.addWeighted(overlay, 0.92, frame, 0.08, 0)
