"""
PHASE 2.3: Confidence-Weighted Plate Selection
===============================================
Smart selection from multiple detection sources with confidence weighting.

Problem:
- Multiple sources (SDK, YOLO, Fortress) may return different plates
- Need to pick the best one, not just highest confidence
- Source reliability varies (SDK more stable, YOLO more flexible)

Solution:
- Weight confidence by source reliability
- Consider plate format validity
- Apply temporal consensus from recent detections
- Return best candidate with combined score
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import time

logger = logging.getLogger(__name__)


@dataclass
class PlateCandidate:
    """A plate detection candidate from any source."""
    plate_text: str
    confidence: float
    source: str  # "dahua_sdk", "yolo", "fortress", "paddleocr"
    timestamp: float = field(default_factory=time.time)
    bbox: Optional[Tuple[int, int, int, int]] = None
    is_valid_format: bool = False
    
    # Computed fields
    weighted_score: float = 0.0
    

# Source reliability weights (based on empirical testing)
SOURCE_WEIGHTS = {
    "dahua_sdk": 0.95,     # SDK has good hardware-accelerated detection
    "fortress": 0.90,      # Fortress pipeline is well-tuned
    "paddleocr": 0.80,     # PaddleOCR general purpose
    "yolo": 0.75,          # YOLO detection, needs OCR
    "easyocr": 0.70,       # EasyOCR as fallback
    "default": 0.60,       # Unknown source
}

# Format validity bonus
FORMAT_VALID_BONUS = 0.15

# Length bonuses (8-9 chars are most common for VN plates)
LENGTH_BONUSES = {
    7: 0.05,
    8: 0.10,
    9: 0.10,
    10: 0.05,
}


from app.services.plate.plate_utils import normalize_plate_basic, is_valid_vn_plate_format


def calculate_weighted_score(candidate: PlateCandidate) -> float:
    """
    Calculate weighted score for a plate candidate.
    
    Factors:
    - Base confidence from OCR/detection
    - Source reliability weight
    - Format validity bonus
    - Length bonus
    """
    # Base score
    score = candidate.confidence
    
    # Apply source weight
    source_weight = SOURCE_WEIGHTS.get(candidate.source.lower(), SOURCE_WEIGHTS["default"])
    score *= source_weight
    
    # Check format validity
    clean_text = normalize_plate_basic(candidate.plate_text)
    candidate.is_valid_format = is_valid_vn_plate_format(clean_text)
    
    if candidate.is_valid_format:
        score += FORMAT_VALID_BONUS
    
    # Length bonus
    text_len = len(clean_text)
    if text_len in LENGTH_BONUSES:
        score += LENGTH_BONUSES[text_len]
    
    # Cap at 1.0
    candidate.weighted_score = min(1.0, score)
    
    return candidate.weighted_score


class PlateSelector:
    """
    Phase 2.3: Intelligent plate selection from multiple sources.
    
    Usage:
        selector = PlateSelector()
        
        # Add candidates from different sources
        selector.add_candidate("29A12345", 0.85, "dahua_sdk")
        selector.add_candidate("29A12346", 0.90, "paddleocr")
        selector.add_candidate("29A12345", 0.75, "paddleocr")
        
        # Get best plate
        best = selector.get_best_plate()
        print(f"Best: {best.plate_text} (score={best.weighted_score:.2f})")
    """
    
    def __init__(self, temporal_window_seconds: float = 5.0):
        """
        Args:
            temporal_window_seconds: Time window for considering recent detections
        """
        self.temporal_window = temporal_window_seconds
        self.candidates: List[PlateCandidate] = []
        self._history: Dict[str, List[PlateCandidate]] = defaultdict(list)
    
    def add_candidate(self, 
                      plate_text: str, 
                      confidence: float, 
                      source: str,
                      bbox: Optional[Tuple[int, int, int, int]] = None,
                      camera_id: str = "default") -> PlateCandidate:
        """
        Add a plate detection candidate.
        
        Args:
            plate_text: Detected plate text
            confidence: OCR/detection confidence (0-1)
            source: Detection source name
            bbox: Optional bounding box
            camera_id: Camera identifier for temporal tracking
        
        Returns:
            PlateCandidate with weighted score
        """
        candidate = PlateCandidate(
            plate_text=plate_text,
            confidence=confidence,
            source=source,
            timestamp=time.time(),
            bbox=bbox
        )
        
        # Calculate weighted score
        calculate_weighted_score(candidate)
        
        # Add to candidates
        self.candidates.append(candidate)
        
        # Add to history for temporal consensus
        self._history[camera_id].append(candidate)
        self._cleanup_history(camera_id)
        
        logger.debug(
            f"Added candidate: '{plate_text}' from {source}, "
            f"conf={confidence:.2f}, weighted={candidate.weighted_score:.2f}"
        )
        
        return candidate
    
    def get_best_plate(self, 
                       min_confidence: float = 0.65,
                       require_valid_format: bool = True,
                       camera_id: str = "default",
                       auto_clear: bool = True) -> Optional[PlateCandidate]:
        """
        Get the best plate from all candidates.
        
        Args:
            min_confidence: Minimum weighted score to accept
            require_valid_format: Only return valid VN format plates
        
        Returns:
            Best PlateCandidate or None
        """
        if not self.candidates:
            return None
        
        # Filter candidates
        valid_candidates = [
            c for c in self.candidates
            if c.weighted_score >= min_confidence
        ]
        
        if require_valid_format:
            valid_candidates = [c for c in valid_candidates if c.is_valid_format]
        
        if not valid_candidates:
            if auto_clear:
                self.clear()
            return None
        
        # Apply temporal consensus bonus
        self._apply_consensus_bonus(valid_candidates, camera_id=camera_id)
        
        # Sort by weighted score
        valid_candidates.sort(key=lambda c: c.weighted_score, reverse=True)
        
        best = valid_candidates[0]
        if auto_clear:
            self.clear()
        return best
    
    def get_consensus_plate(self, camera_id: str = "default") -> Optional[Tuple[str, float]]:
        """
        Get plate text that appears most frequently in recent history.
        
        Returns:
            (plate_text, consensus_ratio) or None
        """
        history = self._history.get(camera_id, [])
        
        if not history:
            return None
        
        # Count normalized plates
        plate_counts: Dict[str, int] = defaultdict(int)
        plate_to_original: Dict[str, str] = {}
        
        for c in history:
            normalized = normalize_plate_basic(c.plate_text)
            plate_counts[normalized] += 1
            plate_to_original[normalized] = c.plate_text
        
        if not plate_counts:
            return None
        
        # Get winner
        best_norm = max(plate_counts, key=plate_counts.get)
        best_count = plate_counts[best_norm]
        consensus_ratio = best_count / len(history)
        
        return plate_to_original[best_norm], consensus_ratio
    
    def _apply_consensus_bonus(self, candidates: List[PlateCandidate], camera_id: str):
        """Apply bonus to candidates that match recent history."""
        # Get recent plates (single camera)
        recent_plates = []
        now = time.time()
        
        history = self._history.get(camera_id, [])
        for c in history:
            if now - c.timestamp <= self.temporal_window:
                recent_plates.append(c.plate_text.upper())
        
        if not recent_plates:
            return
        
        # Count occurrences
        plate_counts = defaultdict(int)
        for p in recent_plates:
            normalized = normalize_plate_basic(p)
            plate_counts[normalized] += 1
        
        # Apply bonus to matching candidates
        for c in candidates:
            normalized = normalize_plate_basic(c.plate_text)
            count = plate_counts.get(normalized, 0)
            
            if count >= 2:
                # Consensus bonus: up to 0.1 for high agreement
                bonus = min(0.10, count * 0.02)
                c.weighted_score = min(1.0, c.weighted_score + bonus)
                logger.debug(f"Consensus bonus +{bonus:.2f} for '{c.plate_text}' (count={count})")
    
    def _cleanup_history(self, camera_id: str):
        """Remove old entries from history."""
        now = time.time()
        cutoff = now - self.temporal_window * 2
        
        self._history[camera_id] = [
            c for c in self._history[camera_id]
            if c.timestamp > cutoff
        ]
    
    def clear(self):
        """Clear all candidates (for new detection batch)."""
        self.candidates.clear()
    
    def clear_history(self, camera_id: str = None):
        """Clear history for camera or all cameras."""
        if camera_id:
            self._history[camera_id].clear()
        else:
            self._history.clear()

    def evict_idle_cameras(self, max_idle_seconds: float = 60.0) -> int:
        """Remove history for cameras that haven't sent data recently.

        Returns:
            Number of cameras evicted.
        """
        now = time.time()
        idle_keys = [
            cam_id for cam_id, entries in self._history.items()
            if not entries or (now - entries[-1].timestamp) > max_idle_seconds
        ]
        for key in idle_keys:
            del self._history[key]
        return len(idle_keys)


def select_best_plate(candidates: List[Tuple[str, float, str]], 
                      min_confidence: float = 0.65,
                      camera_id: str = "default",
                      auto_clear: bool = True) -> Optional[Tuple[str, float]]:
    """
    Convenience function to select best plate from list.
    
    Args:
        candidates: List of (plate_text, confidence, source) tuples
        min_confidence: Minimum weighted score
    
    Returns:
        (best_plate_text, best_score) or None
    """
    selector = PlateSelector()
    
    for text, conf, source in candidates:
        selector.add_candidate(text, conf, source)
    
    best = selector.get_best_plate(min_confidence=min_confidence, camera_id=camera_id, auto_clear=auto_clear)
    
    if best:
        return best.plate_text, best.weighted_score
    
    return None
