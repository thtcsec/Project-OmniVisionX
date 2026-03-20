from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from app.services.core.enhancer import ImageEnhancer


def _to_rgb_array(img: Image.Image) -> np.ndarray:
    arr = np.array(img)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    return arr


def apply_night_style(img_bgr: np.ndarray, strength: float = 0.7) -> np.ndarray:
    strength = float(max(0.0, min(1.0, strength)))
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    enhanced = ImageEnhancer.preprocess_night_plate(pil)
    enhanced_rgb = _to_rgb_array(enhanced).astype(np.float32)
    base_rgb = rgb.astype(np.float32)
    blended = base_rgb * (1.0 - strength) + enhanced_rgb * strength
    noise = np.random.RandomState(hash(img_bgr.tobytes()[:64]) % (2**31)).normal(0, 8 + 8 * strength, blended.shape).astype(np.float32)
    blended = np.clip(blended + noise, 0, 255)
    gamma = 1.0 + 0.2 * strength
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
    corrected = cv2.LUT(blended.astype(np.uint8), lut)
    return cv2.cvtColor(corrected, cv2.COLOR_RGB2BGR)

def apply_ir_style(img_bgr: np.ndarray, strength: float = 0.7) -> np.ndarray:
    strength = float(max(0.0, min(1.0, strength)))
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if np.random.random() < 0.5:
        gray = 255 - gray
    gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32)
    base = img_bgr.astype(np.float32)
    blended = base * (1.0 - strength) + gray_rgb * strength
    return np.clip(blended, 0, 255).astype(np.uint8)

def apply_haze_style(img_bgr: np.ndarray, strength: float = 0.6) -> np.ndarray:
    strength = float(max(0.0, min(1.0, strength)))
    haze = img_bgr.astype(np.float32)
    veil = np.full_like(haze, 200.0)
    blended = haze * (1.0 - strength) + veil * strength
    blended = cv2.GaussianBlur(blended, (0, 0), sigmaX=1.2 + 2.0 * strength)
    return np.clip(blended, 0, 255).astype(np.uint8)

def apply_lowres_style(img_bgr: np.ndarray, scale: float = 0.5, jpeg_quality: Optional[int] = None) -> np.ndarray:
    scale = float(max(0.1, min(1.0, scale)))
    h, w = img_bgr.shape[:2]
    dw = max(8, int(w * scale))
    dh = max(8, int(h * scale))
    small = cv2.resize(img_bgr, (dw, dh), interpolation=cv2.INTER_AREA)
    up = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    if jpeg_quality is None:
        return up
    quality = int(max(30, min(95, jpeg_quality)))
    ok, enc = cv2.imencode(".jpg", up, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return up
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return dec if dec is not None else up


def bridge_dataset(input_dir: str,
                   output_dir: str,
                   mode: str = "night",
                   strength: float = 0.7,
                   max_images: Optional[int] = None,
                   lowres_scale: Optional[float] = None,
                   jpeg_quality: Optional[int] = None,
                   seed: Optional[int] = 42) -> Tuple[int, int]:
    in_dir = Path(input_dir)
    images_dir = in_dir / "images" if (in_dir / "images").exists() else in_dir
    labels_dir = in_dir / "labels" if (in_dir / "labels").exists() else in_dir

    out_dir = Path(output_dir)
    out_images = out_dir / "images"
    out_labels = out_dir / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    # Seed RNG for reproducible augmentation
    rng = np.random.RandomState(seed)

    image_paths = [p for p in images_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    total = 0
    bridged = 0
    for img_path in image_paths:
        if max_images is not None and bridged >= max_images:
            break
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        total += 1
        if mode == "mix":
            current_mode = rng.choice(["night", "ir", "lowres", "haze"])
        else:
            current_mode = mode
        if current_mode == "night":
            img = apply_night_style(img, strength=strength)
        elif current_mode == "ir":
            img = apply_ir_style(img, strength=strength)
        elif current_mode == "haze":
            img = apply_haze_style(img, strength=strength)
        elif current_mode == "lowres":
            img = apply_lowres_style(img, scale=lowres_scale or 0.5, jpeg_quality=jpeg_quality)
        out_path = out_images / img_path.name
        cv2.imwrite(str(out_path), img)

        label_path = labels_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            out_label = out_labels / label_path.name
            out_label.write_text(label_path.read_text(encoding="utf-8"), encoding="utf-8")
        bridged += 1

    return total, bridged
