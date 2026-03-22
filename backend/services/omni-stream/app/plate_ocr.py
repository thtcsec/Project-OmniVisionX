"""PaddleOCR wrapper for optional plate extraction."""

import logging
import re
from typing import Optional

log = logging.getLogger("omni-stream.ocr")
PLATE_PATTERN = re.compile(r"\d{2}[A-Z]\d?[-.\s]?\d{3,5}[.\s]?\d{0,2}")


class PlateOCR:
    def __init__(self):
        self._ocr = None
        self._available = False
        try:
            from paddleocr import PaddleOCR

            self._ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False, use_gpu=False)
            self._available = True
            log.info("PaddleOCR initialized OK")
        except Exception as exc:
            log.warning("PaddleOCR not available, plate OCR disabled: %s", exc)

    def read_plate(self, crop) -> Optional[str]:
        if not self._available or crop is None or crop.size == 0:
            return None
        try:
            results = self._ocr.ocr(crop, cls=True)
            if not results or not results[0]:
                return None

            texts = []
            for line in results[0]:
                if not line or len(line) < 2:
                    continue
                text_info = line[1]
                if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                    text = str(text_info[0]).strip()
                    conf = float(text_info[1])
                else:
                    text = str(text_info).strip()
                    conf = 0.0
                if conf > 0.5:
                    texts.append(text)

            combined = "".join(texts).upper()
            if not combined:
                return None

            match = PLATE_PATTERN.search(combined)
            if match:
                return match.group(0)

            cleaned = re.sub(r"[^A-Z0-9\-.]", "", combined)
            if 6 <= len(cleaned) <= 12:
                return cleaned
            return None
        except Exception as exc:
            log.debug("OCR error: %s", exc)
            return None
