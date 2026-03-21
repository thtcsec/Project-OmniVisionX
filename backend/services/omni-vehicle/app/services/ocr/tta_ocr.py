"""
PHASE 2.1: Test-Time Augmentation (TTA) for OCR
================================================
Apply multiple augmentations at inference time and vote on results.

Problem:
- Single OCR pass may miss characters due to lighting/angle
- Glare, shadows, motion blur affect recognition
- Model uncertainty not captured

Solution:
- Apply 4-8 augmentations per plate image
- Run OCR on each augmented version
- Weighted voting based on confidence + consistency
- Return consensus result with boosted confidence

Augmentations:
- Brightness +/- 20%
- Contrast +/- 15%
- Gamma 0.8/1.2
- Slight rotation +/- 3 degrees
- Horizontal flip (for symmetric checking)
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional, Callable, Dict
from dataclasses import dataclass
from collections import defaultdict
import re

logger = logging.getLogger(__name__)


@dataclass
class TTAResult:
    """Result from TTA OCR"""
    plate_text: str
    confidence: float
    vote_count: int
    total_votes: int
    augmentations_used: int
    raw_results: List[Tuple[str, float]]  # (text, conf) per augmentation


class TTAAugmentor:
    """
    Apply test-time augmentations to plate images.
    4 targeted passes thay vì 6-8 random — đủ để cover các trường hợp thực tế
    mà không tốn gấp đôi CPU.
    """

    def __init__(self, num_augmentations: int = 4):
        """
        Args:
            num_augmentations: Number of augmented versions to create (default 4)
        """
        self.num_augmentations = num_augmentations

    def generate_augmentations(self, img: np.ndarray) -> List[Tuple[np.ndarray, str]]:
        """
        Generate targeted augmented versions of input image.
        4 passes covering the most common plate degradation scenarios:
        - Original (baseline)
        - CLAHE adaptive (low-contrast plates)
        - Unsharp mask (motion blur / slightly soft)
        - Gamma 0.8 (overexposed / glare from retroreflective plates)

        Returns:
            List of (augmented_image, augmentation_name)
        """
        augmented = []

        # 0. Original — always include
        augmented.append((img.copy(), "original"))

        # 1. CLAHE adaptive — handles low-contrast, faded plates
        clahe = self._apply_clahe(img)
        augmented.append((clahe, "clahe"))

        # 2. Unsharp mask — recovers slightly soft/blurred characters
        sharp = self._unsharp_mask(img)
        augmented.append((sharp, "unsharp"))

        # 3. Gamma 0.8 — reduces glare on overexposed retroreflective plates
        gamma_low = self._adjust_gamma(img, 0.8)
        augmented.append((gamma_low, "gamma0.8"))

        return augmented[:self.num_augmentations]
    
    def _adjust_brightness(self, img: np.ndarray, factor: float) -> np.ndarray:
        """Adjust image brightness by factor."""
        return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    def _adjust_contrast(self, img: np.ndarray, factor: float) -> np.ndarray:
        """Adjust image contrast by factor."""
        mean = np.mean(img)
        return np.clip((img.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)

    def _adjust_gamma(self, img: np.ndarray, gamma: float) -> np.ndarray:
        """Apply gamma correction."""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in np.arange(0, 256)]).astype(np.uint8)
        return cv2.LUT(img, table)

    def _apply_clahe(self, img: np.ndarray) -> np.ndarray:
        """Apply CLAHE enhancement — handles low-contrast / faded plates."""
        if len(img.shape) == 3:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(img)

    def _sharpen(self, img: np.ndarray) -> np.ndarray:
        """Apply sharpening kernel."""
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        return cv2.filter2D(img, -1, kernel)

    def _unsharp_mask(self, img: np.ndarray, sigma: float = 1.0, strength: float = 1.5) -> np.ndarray:
        """Unsharp mask — better than simple sharpen for motion-blurred plates."""
        blurred = cv2.GaussianBlur(img, (0, 0), sigma)
        return np.clip(img.astype(np.float32) + strength * (img.astype(np.float32) - blurred.astype(np.float32)), 0, 255).astype(np.uint8)


class TTAOCR:
    """
    Phase 2.1: Test-Time Augmentation OCR.

    Runs OCR on 4 targeted augmented versions and votes on result.
    4 passes thay vì 6-8 — tiết kiệm ~40% CPU, coverage tốt hơn nhờ targeted.

    Usage:
        tta = TTAOCR(ocr_func=paddle_ocr.ocr_single)
        result = tta.recognize(plate_image)
        print(f"Plate: {result.plate_text}, Conf: {result.confidence:.2f}")
    """

    def __init__(self,
                 ocr_func: Callable[[np.ndarray], Tuple[str, float]],
                 num_augmentations: int = 4,
                 min_confidence: float = 0.3):
        """
        Args:
            ocr_func: Function that takes image and returns (text, confidence)
            num_augmentations: Number of augmented versions to process (default 4)
            min_confidence: Minimum confidence to include in voting
        """
        self.ocr_func = ocr_func
        self.augmentor = TTAAugmentor(num_augmentations)
        self.min_confidence = min_confidence
    
    def recognize(self, plate_img: np.ndarray) -> TTAResult:
        """
        Run TTA OCR on plate image.
        
        Args:
            plate_img: Plate crop image (BGR or grayscale)
        
        Returns:
            TTAResult with consensus plate text and confidence
        """
        if plate_img is None or plate_img.size == 0:
            return TTAResult(
                plate_text="",
                confidence=0.0,
                vote_count=0,
                total_votes=0,
                augmentations_used=0,
                raw_results=[]
            )
        
        # Generate augmentations
        augmented_images = self.augmentor.generate_augmentations(plate_img)
        
        # Run OCR on each
        raw_results = []
        for aug_img, aug_name in augmented_images:
            try:
                text, conf = self.ocr_func(aug_img)
                if text and conf >= self.min_confidence:
                    raw_results.append((text, conf, aug_name))
                    logger.debug(f"TTA {aug_name}: '{text}' (conf={conf:.2f})")
            except Exception as e:
                logger.warning(f"TTA OCR failed for {aug_name}: {e}")
        
        if not raw_results:
            return TTAResult(
                plate_text="",
                confidence=0.0,
                vote_count=0,
                total_votes=len(augmented_images),
                augmentations_used=len(augmented_images),
                raw_results=[]
            )
        
        # Vote on results
        return self._vote_results(raw_results, len(augmented_images))
    
    def _vote_results(self, results: List[Tuple[str, float, str]], 
                      total_augs: int) -> TTAResult:
        """
        Weighted voting on OCR results with outlier protection.
        
        Fixes:
        - Cap individual vote weights to prevent single high-confidence outlier dominance
        - Use median confidence for more robust consensus
        - Weight by both confidence and vote count
        """
        # Normalize texts for voting
        votes: Dict[str, List[float]] = defaultdict(list)  # Store all confidences per text
        text_to_original: Dict[str, str] = {}
        
        for text, conf, aug_name in results:
            normalized = self._normalize_for_voting(text)
            votes[normalized].append(conf)
            
            # Keep the original text with the HIGHEST confidence for each normalized form
            if normalized not in text_to_original or conf > max(votes[normalized][:-1], default=0.0):
                text_to_original[normalized] = text
        
        if not votes:
            return TTAResult(
                plate_text="",
                confidence=0.0,
                vote_count=0,
                total_votes=total_augs,
                augmentations_used=total_augs,
                raw_results=[(r[0], r[1]) for r in results]
            )
        
        # Calculate weighted scores with outlier protection
        weighted_scores: Dict[str, float] = {}
        
        for normalized, confidences in votes.items():
            vote_count = len(confidences)
            
            # Use median confidence to reduce outlier impact
            median_conf = np.median(confidences)
            
            # Cap individual confidence contributions (max 0.9 per vote)
            capped_confs = [min(0.9, conf) for conf in confidences]
            mean_capped_conf = np.mean(capped_confs)
            
            # Weighted score: (vote_count_weight * median_conf_weight)
            vote_weight = vote_count / len(results)  # Fraction of total votes
            conf_weight = (median_conf + mean_capped_conf) / 2  # Balanced confidence
            
            # Final score balances consensus and confidence
            weighted_scores[normalized] = vote_weight * 0.6 + conf_weight * 0.4
            
            logger.debug(
                f"TTA candidate '{normalized}': {vote_count} votes, "
                f"median_conf={median_conf:.2f}, score={weighted_scores[normalized]:.3f}"
            )
        
        # Find winner
        winner_norm = max(weighted_scores, key=weighted_scores.get)
        winner_score = weighted_scores[winner_norm]
        winner_text = text_to_original[winner_norm]
        winner_confidences = votes[winner_norm]
        
        # Count actual votes for winner
        vote_count = len(winner_confidences)
        
        # Calculate final confidence using median (more robust than mean)
        median_conf = np.median(winner_confidences)
        consensus_ratio = vote_count / len(results)
        
        # Boost confidence based on consensus strength
        if consensus_ratio >= 0.8 and vote_count >= 3:
            boosted_conf = min(1.0, median_conf * 1.15)  # Strong consensus
        elif consensus_ratio >= 0.6 and vote_count >= 2:
            boosted_conf = min(1.0, median_conf * 1.05)  # Good consensus
        elif vote_count == 1 and median_conf > 0.85:
            boosted_conf = median_conf * 0.9  # Single high-conf vote penalty
        else:
            boosted_conf = median_conf * 0.95  # Slight penalty for low consensus
        
        logger.debug(
            f"TTA Vote: '{winner_text}' won with {vote_count}/{len(results)} votes, "
            f"median_conf={median_conf:.2f}→{boosted_conf:.2f}, score={winner_score:.3f}"
        )
        
        return TTAResult(
            plate_text=winner_text,
            confidence=boosted_conf,
            vote_count=vote_count,
            total_votes=len(results),
            augmentations_used=total_augs,
            raw_results=[(r[0], r[1]) for r in results]
        )
    
    def _normalize_for_voting(self, text: str) -> str:
        """Normalize plate text for voting comparison."""
        # Remove non-alphanumeric, uppercase
        normalized = re.sub(r'[^A-Z0-9]', '', text.upper())
        return normalized


def create_tta_ocr(ocr_func: Callable, num_augmentations: int = 4) -> TTAOCR:
    """Factory function to create TTA OCR instance. Default 4 targeted passes."""
    return TTAOCR(ocr_func=ocr_func, num_augmentations=num_augmentations)
