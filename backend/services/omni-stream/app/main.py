"""omni-stream - unified detection + server-side overlay MJPEG streamer."""

import logging
import os
import threading
import time
import uuid
from datetime import datetime, timezone

import cv2
from flask import Flask, Response
from ultralytics import YOLO

from overlay import draw_detections
from plate_ocr import PlateOCR
from publisher import EventPublisher
from tracker import SimpleByteTrack

RTSP_URL = os.getenv(
    "CAMERA_RTSP_URL",
    "rtsp://admin:password@192.168.1.100:554/cam/realmonitor?channel=1&subtype=0",
)
MODEL_PATH = os.getenv("OMNI_MODEL_PATH", "yolov8n.pt")
CONFIDENCE = float(os.getenv("OMNI_CONFIDENCE", "0.4"))
STREAM_PORT = int(os.getenv("STREAM_OUTPUT_PORT", "8090"))
FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", "1280"))
FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", "720"))
TARGET_FPS = int(os.getenv("TARGET_FPS", "15"))
CAMERA_ID = os.getenv("CAMERA_ID", "cam-01")

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("omni-stream")

app = Flask(__name__)
latest_frame: bytes = b""
frame_lock = threading.Lock()


def detection_loop():
    global latest_frame

    log.info("Loading YOLO model: %s", MODEL_PATH)
    model = YOLO(MODEL_PATH)
    tracker = SimpleByteTrack()
    ocr = PlateOCR()
    publisher = EventPublisher()
    frame_interval = 1.0 / max(1, TARGET_FPS)

    while True:
        cap = cv2.VideoCapture(RTSP_URL)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            log.warning("Cannot open RTSP: %s - retrying in 3s", RTSP_URL)
            time.sleep(3)
            continue

        log.info("RTSP connected: %s", RTSP_URL)
        while cap.isOpened():
            loop_start = time.monotonic()
            ret, frame = cap.read()
            if not ret:
                log.warning("Frame grab failed, reconnecting")
                break

            grab_ts = datetime.now(timezone.utc).isoformat()
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            h, w = frame.shape[:2]

            results = model.predict(frame, conf=CONFIDENCE, verbose=False, imgsz=640)
            detections = []
            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                names = results[0].names if hasattr(results[0], "names") else model.names
                for idx in range(len(boxes)):
                    xyxy = boxes.xyxy[idx].cpu().numpy().astype(int)
                    conf = float(boxes.conf[idx].cpu().numpy())
                    cls_id = int(boxes.cls[idx].cpu().numpy())
                    label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
                    detections.append(
                        {
                            "bbox": xyxy.tolist(),
                            "confidence": round(conf, 3),
                            "label": label,
                            "cls_id": cls_id,
                        }
                    )

            tracked = tracker.update(detections, (w, h))

            for det in tracked:
                if str(det.get("label", "")).lower() in VEHICLE_CLASSES:
                    x1, y1, x2, y2 = det["bbox"]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        plate = ocr.read_plate(crop)
                        if plate:
                            det["plate_number"] = plate

            annotated = draw_detections(frame, tracked)
            ok, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                with frame_lock:
                    latest_frame = jpeg.tobytes()

            for det in tracked:
                publisher.publish(
                    {
                        "event_id": str(uuid.uuid4()),
                        "camera_id": CAMERA_ID,
                        "service": "omni-stream",
                        "event_type": "detection",
                        "label": det.get("label"),
                        "bbox": det.get("bbox"),
                        "confidence": det.get("confidence"),
                        "track_id": det.get("track_id"),
                        "plate_number": det.get("plate_number"),
                        "frame_width": w,
                        "frame_height": h,
                        "timestamp": grab_ts,
                    }
                )

            elapsed = time.monotonic() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        cap.release()
        time.sleep(1)


def generate_mjpeg():
    while True:
        with frame_lock:
            frame_data = latest_frame
        if frame_data:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_data + b"\r\n"
        time.sleep(1.0 / (max(1, TARGET_FPS) + 5))


@app.route("/stream")
def stream():
    return Response(generate_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/health")
def health():
    return {"status": "ok", "service": "omni-stream"}


if __name__ == "__main__":
    log.info("Starting detection loop")
    worker = threading.Thread(target=detection_loop, daemon=True)
    worker.start()

    log.info("Starting MJPEG server on port %d", STREAM_PORT)
    app.run(host="0.0.0.0", port=STREAM_PORT, threaded=True)
