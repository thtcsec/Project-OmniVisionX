import logging
import os
import uuid
from datetime import datetime

import cv2
import numpy as np

logger = logging.getLogger("omni-vehicle.stream_thumbnail")
try:
    from app.services.storage.s3_uploader import try_build_uploader
except Exception:
    try_build_uploader = None


class LprThumbnailService:
    def __init__(self, settings):
        self.settings = settings
        self._uploader = try_build_uploader(settings) if try_build_uploader else None

    @staticmethod
    def _derive_vehicle_bbox_from_plate(
        frame_shape: tuple[int, int, int] | tuple[int, int],
        vehicle_bbox: tuple | list | None,
        plate_bbox: tuple | list | None,
    ) -> tuple[int, int, int, int] | None:
        """Derive a vehicle bounding box for drawing on the full-frame image.

        Priority:
        1. Use the YOLO vehicle_bbox directly (already in frame coordinates)
        2. Fall back to expanding plate_bbox if no vehicle_bbox
        """
        frame_h, frame_w = frame_shape[:2]

        # --- Priority 1: Use vehicle_bbox from YOLO detection ---
        if vehicle_bbox and len(vehicle_bbox) == 4:
            try:
                vx1, vy1, vx2, vy2 = [int(v) for v in vehicle_bbox]
                # Reject all-zero bbox (placeholder from missing Redis field)
                if not (vx1 == 0 and vy1 == 0 and vx2 == 0 and vy2 == 0):
                    vx1 = max(0, min(frame_w - 1, vx1))
                    vy1 = max(0, min(frame_h - 1, vy1))
                    vx2 = max(0, min(frame_w - 1, vx2))
                    vy2 = max(0, min(frame_h - 1, vy2))
                    if vx2 > vx1 and vy2 > vy1:
                        return (vx1, vy1, vx2, vy2)
            except Exception:
                pass

        # --- Priority 2: ALWAYS expand from plate_bbox as fallback ---
        if plate_bbox and len(plate_bbox) == 4:
            try:
                px1, py1, px2, py2 = [int(v) for v in plate_bbox]
                # Reject degenerate plate bbox (all zeros or inverted)
                if px1 == 0 and py1 == 0 and px2 == 0 and py2 == 0:
                    pass  # fall through to return None
                else:
                    px1 = max(0, min(frame_w - 1, px1))
                    py1 = max(0, min(frame_h - 1, py1))
                    px2 = max(0, min(frame_w - 1, px2))
                    py2 = max(0, min(frame_h - 1, py2))
                    if px2 > px1 and py2 > py1:
                        plate_w = max(1, px2 - px1)
                        plate_h = max(1, py2 - py1)
                        expand_x = max(30, int(plate_w * 2.8))
                        expand_top = max(40, int(plate_h * 4.5))
                        expand_bottom = max(40, int(plate_h * 2.2))
                        vx1 = max(0, px1 - expand_x)
                        vy1 = max(0, py1 - expand_top)
                        vx2 = min(frame_w - 1, px2 + expand_x)
                        vy2 = min(frame_h - 1, py2 + expand_bottom)
                        if vx2 > vx1 and vy2 > vy1:
                            return (vx1, vy1, vx2, vy2)
            except Exception:
                pass

        return None

    def save_thumbnails(
        self,
        full_frame: np.ndarray,
        plate_crop: np.ndarray | None,
        camera_id: str,
        plate_text: str,
        plate_bbox: tuple | None = None,
        vehicle_bbox: tuple | list | None = None,
        vehicle_type: str | None = None,
        vehicle_color: str | None = None,
    ) -> tuple:
        try:
            if full_frame is None or not hasattr(full_frame, "shape") or full_frame.size == 0:
                return "", ""
            thumb_dir = self.settings.thumbnail_path
            os.makedirs(thumb_dir, exist_ok=True)
            plate_jpeg_quality = int(getattr(self.settings, "lpr_plate_jpeg_quality", 98))
            frame_jpeg_quality = int(getattr(self.settings, "lpr_frame_jpeg_quality", 90))
            plate_jpeg_quality = max(70, min(100, plate_jpeg_quality))
            frame_jpeg_quality = max(65, min(100, frame_jpeg_quality))
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_plate = "".join(c for c in plate_text if c.isalnum()) or "NOPLATE"
            uid = uuid.uuid4().hex[:6]
            base = f"{ts}_{camera_id[:8]}_{safe_plate}_{uid}"
            plate_path = os.path.join(thumb_dir, f"{base}_plate.jpg")
            frame_path = os.path.join(thumb_dir, f"{base}_full.jpg")

            plate_saved = False
            frame_saved = False
            plate_basename = ""
            frame_basename = ""

            if plate_crop is not None and plate_crop.size != 0:
                display_crop = plate_crop.copy()
                crop_h, crop_w = display_crop.shape[:2]
                if crop_w > 0 and crop_h > 0:
                    try:
                        gray = cv2.cvtColor(display_crop, cv2.COLOR_BGR2GRAY)
                        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                    except Exception:
                        sharpness = 0.0
                    min_display_width = 384 if sharpness < 120.0 else 320
                    if crop_w < min_display_width:
                        scale = min_display_width / crop_w
                        display_crop = cv2.resize(display_crop, (int(crop_w * scale), int(crop_h * scale)), interpolation=cv2.INTER_LANCZOS4)
                    scaled_h, scaled_w = display_crop.shape[:2]
                    if scaled_h < 96:
                        extra_scale = 96.0 / max(1, scaled_h)
                        display_crop = cv2.resize(display_crop, (int(scaled_w * extra_scale), int(scaled_h * extra_scale)), interpolation=cv2.INTER_LANCZOS4)
                    display_crop = cv2.bilateralFilter(display_crop, 5, 30, 30)
                    lab = cv2.cvtColor(display_crop, cv2.COLOR_BGR2LAB)
                    l_channel, a_channel, b_channel = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))
                    l_channel = clahe.apply(l_channel)
                    display_crop = cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)
                    sharpen_sigma = 0.9 if sharpness >= 120.0 else 1.1
                    sharpen_gain = 1.16 if sharpness >= 120.0 else 1.24
                    blurred = cv2.GaussianBlur(display_crop, (0, 0), sharpen_sigma)
                    display_crop = cv2.addWeighted(display_crop, sharpen_gain, blurred, -(sharpen_gain - 1.0), 0)
                plate_saved = cv2.imwrite(plate_path, display_crop, [cv2.IMWRITE_JPEG_QUALITY, plate_jpeg_quality])
                if plate_saved:
                    plate_basename = os.path.basename(plate_path)

            review_frame = full_frame.copy()
            frame_h, frame_w = review_frame.shape[:2]
            thickness = max(2, int(min(frame_h, frame_w) / 360))

            final_vehicle_bbox = self._derive_vehicle_bbox_from_plate(
                review_frame.shape,
                vehicle_bbox=vehicle_bbox,
                plate_bbox=plate_bbox,
            )
            if final_vehicle_bbox is None:
                logger.warning(
                    "Cannot derive vehicle bbox for %s: vehicle_bbox=%s plate_bbox=%s frame=%dx%d",
                    plate_text, vehicle_bbox, plate_bbox, frame_w, frame_h,
                )
            if final_vehicle_bbox is not None:
                vx1, vy1, vx2, vy2 = final_vehicle_bbox
                box_thickness = max(thickness, 3)
                cv2.rectangle(review_frame, (vx1, vy1), (vx2, vy2), (46, 204, 113), box_thickness)
                label_parts = []
                if vehicle_type and vehicle_type != "unknown":
                    label_parts.append(vehicle_type)
                if vehicle_color and vehicle_color != "unknown":
                    label_parts.append(vehicle_color)
                label = " | ".join(label_parts).upper() if label_parts else ""
                if label:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.52
                    text_thickness = max(1, thickness - 1)
                    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
                    tx1 = vx1
                    ty1 = max(0, vy1 - text_h - baseline - 8)
                    tx2 = min(frame_w - 1, tx1 + text_w + 8)
                    ty2 = max(0, vy1 - 2)
                    cv2.rectangle(review_frame, (tx1, ty1), (tx2, ty2), (46, 204, 113), -1)
                    cv2.putText(review_frame, label, (tx1 + 4, ty2 - 4), font, font_scale, (10, 15, 15), text_thickness, cv2.LINE_AA)

            if plate_bbox is not None and len(plate_bbox) == 4:
                px1, py1, px2, py2 = [int(v) for v in plate_bbox]
                px1 = max(0, min(frame_w - 1, px1))
                py1 = max(0, min(frame_h - 1, py1))
                px2 = max(0, min(frame_w - 1, px2))
                py2 = max(0, min(frame_h - 1, py2))
                if px2 > px1 and py2 > py1:
                    cv2.rectangle(review_frame, (px1, py1), (px2, py2), (0, 0, 255), thickness)

            frame_saved = cv2.imwrite(frame_path, review_frame, [cv2.IMWRITE_JPEG_QUALITY, frame_jpeg_quality])
            if frame_saved:
                frame_basename = os.path.basename(frame_path)

            if not plate_saved and not frame_saved:
                return "", ""
            if self._uploader:
                try:
                    plate_ref = plate_basename
                    frame_ref = frame_basename
                    if plate_basename:
                        key = f"plates/{camera_id}/{plate_basename}"
                        with open(os.path.join(thumb_dir, plate_basename), "rb") as f:
                            _, url = self._uploader.put_jpeg_and_url(key, f.read())
                        if url:
                            plate_ref = url
                    if frame_basename:
                        key = f"frames/{camera_id}/{frame_basename}"
                        with open(os.path.join(thumb_dir, frame_basename), "rb") as f:
                            _, url = self._uploader.put_jpeg_and_url(key, f.read())
                        if url:
                            frame_ref = url
                    return plate_ref, frame_ref
                except Exception as e:
                    logger.warning("Failed to upload to MinIO: %s", e)
            return plate_basename, frame_basename
        except Exception as e:
            logger.warning("Failed to save thumbnails: %s", e)
            return "", ""
