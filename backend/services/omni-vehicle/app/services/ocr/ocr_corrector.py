"""
Vietnamese License Plate OCR Corrector
Context-aware correction without hardcoded if/else rules

Based on:
- Thông tư 58/2020/TT-BCA (VN license plate format regulations)
- OCR confusion matrix (character similarity patterns)
- Structural analysis (position-based validation)
"""
import re
import logging
from typing import Tuple, List, Dict, Set
from collections import Counter
from app.services.plate.plate_utils import normalize_plate_basic
from app.services.plate.plate_constants import (
    VALID_PROVINCE_CODES, VALID_SERIES_LETTERS as _SHARED_SERIES,
    SERIAL_DIGIT_LENGTH, SERIAL_DIGIT_MIN, SERIAL_DIGIT_MAX,
)

logger = logging.getLogger(__name__)


class VNPlateCorrector:
    """
    Intelligent OCR correction using:
    1. Pattern Analysis (VN plate structure per Thông tư 58/2020/TT-BCA)
    2. Character Similarity Matrix (OCR confusion pairs)
    3. Context Validation (province codes, series letters)
    """
    
    # Valid Vietnamese province codes — imported from plate_constants
    VALID_PROVINCES: Set[int] = VALID_PROVINCE_CODES
    
    # Valid series letters — imported from plate_constants (includes J, W)
    VALID_SERIES: Set[str] = _SHARED_SERIES
    
    # OCR Confusion Matrix with WEIGHTED probabilities
    # Format: {actual_char: {confused_char: probability}}
    # Higher probability = more likely confusion
    CONFUSION_MATRIX: Dict[str, Dict[str, float]] = {
        # Digits ↔ Letters (most common OCR errors)
        '0': {'O': 0.4, 'D': 0.3, 'Q': 0.2, '8': 0.1},
        '1': {'I': 0.4, 'L': 0.3, 'T': 0.15, '6': 0.1, 'l': 0.05},
        '2': {'Z': 0.5, '7': 0.3, '1': 0.2},
        '3': {'8': 0.6, 'B': 0.4},
        '4': {'A': 0.7, 'H': 0.3},
        '5': {'S': 0.5, '6': 0.3, '8': 0.2},
        '6': {'G': 0.4, '5': 0.3, 'b': 0.2, '1': 0.1},
        '7': {'T': 0.3, '1': 0.25, 'J': 0.2, '2': 0.15, 'V': 0.1},
        '8': {'B': 0.5, 'S': 0.2, '3': 0.15, '0': 0.15},
        '9': {'g': 0.6, 'q': 0.4},
        
        # Letters ↔ Letters
        'A': {'H': 0.5, '4': 0.3, 'R': 0.2},
        'B': {'8': 0.6, 'S': 0.3, '3': 0.1},
        'C': {'G': 0.6, 'O': 0.4},
        'D': {'O': 0.5, '0': 0.3, 'Q': 0.2},
        'G': {'C': 0.5, '6': 0.3, 'O': 0.2},
        'H': {'A': 0.4, 'M': 0.3, 'K': 0.2, 'N': 0.1},
        'I': {'1': 0.5, 'L': 0.3, 'T': 0.2},
        'K': {'H': 0.5, 'R': 0.3, 'X': 0.2},
        'L': {'I': 0.5, '1': 0.3, 'T': 0.2},
        'M': {'N': 0.4, 'H': 0.3, 'S': 0.2, 'rn': 0.1},
        'N': {'M': 0.5, 'H': 0.3, 'K': 0.2},
        'O': {'0': 0.5, 'D': 0.3, 'Q': 0.2},
        'P': {'R': 0.6, 'B': 0.4},
        'Q': {'O': 0.5, '0': 0.3, 'D': 0.2},
        'R': {'P': 0.5, 'K': 0.3, 'A': 0.2},
        'S': {'5': 0.5, '8': 0.3, 'M': 0.2},
        'T': {'1': 0.4, 'I': 0.3, '7': 0.2, 'L': 0.1},
        'U': {'V': 0.6, 'Y': 0.4},
        'V': {'U': 0.5, 'Y': 0.3, '7': 0.2},
        'X': {'K': 0.5, 'Y': 0.3, 'H': 0.2},
        'Y': {'V': 0.5, 'U': 0.3, 'X': 0.2},
        'Z': {'2': 0.6, '7': 0.4},
    }
    
    def __init__(self):
        # Build reverse confusion matrix with WEIGHTED probabilities
        # Format: {confused_char: [(actual_char, probability), ...]}
        self._inverse_confusion: Dict[str, List[Tuple[str, float]]] = {}
        for actual, confused_dict in self.CONFUSION_MATRIX.items():
            for confused, prob in confused_dict.items():
                if confused not in self._inverse_confusion:
                    self._inverse_confusion[confused] = []
                self._inverse_confusion[confused].append((actual, prob))
        
        # Sort by probability (highest first) for efficient lookup
        for confused in self._inverse_confusion:
            self._inverse_confusion[confused].sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"✅ VNPlateCorrector initialized with {len(self._inverse_confusion)} weighted confusion patterns")
    
    def correct(self, ocr_text: str, raw_confidence: float = 0.0) -> Tuple[str, float, List[str]]:
        """
        Intelligent correction of OCR text using context-aware analysis.
        
        Args:
            ocr_text: Raw OCR output (may contain errors)
            raw_confidence: OCR engine confidence score (0.0-1.0)
        
        Returns:
            Tuple of:
            - corrected_text: Fixed plate number
            - final_confidence: Adjusted confidence (penalized for corrections)
            - corrections_applied: List of correction descriptions for logging
        """
        if not ocr_text or len(ocr_text) < 7:
            return ocr_text, 0.0, []
        
        # Normalize: uppercase, remove separators
        text = normalize_plate_basic(ocr_text)
        
        corrections: List[str] = []
        
        # STAGE 1: Structure Analysis
        # Determine plate structure: 2 digits (province) + 1-2 letters (series) + 4-6 digits (serial)
        structure = self._analyze_structure(text)
        logger.debug(f"Plate structure: {structure}")
        
        # STAGE 2: Context-based Correction (position-aware)
        # Province code (position 0-1): Must be digits 11-99
        text, province_fixes = self._fix_province_code(text, structure)
        corrections.extend(province_fixes)
        
        # Series code (position 2-3): Must be valid letters
        text, series_fixes = self._fix_series_code(text, structure)
        corrections.extend(series_fixes)
        
        # Serial number (position 4+): Must be all digits
        text, serial_fixes = self._fix_serial_number(text, structure)
        corrections.extend(serial_fixes)
        
        # STAGE 3: Pattern-based Correction
        # Fix multi-character confusions (e.g., "RL" → "R1") based on context
        text, pattern_fixes = self._fix_pattern_confusions(text, structure)
        corrections.extend(pattern_fixes)
        
        # STAGE 4: Confidence Adjustment
        # Penalty: -5% per correction applied
        confidence_penalty = len(corrections) * 0.05
        final_confidence = max(0.0, raw_confidence - confidence_penalty)
        
        # Bonus: +10% if final text matches valid VN plate structure
        if self._validate_structure(text):
            final_confidence = min(1.0, final_confidence + 0.1)
        
        # REDUCED LOGGING: Only log if corrections were applied (avoid noise at 30 FPS)
        if corrections:
            logger.debug(f"🔧 OCR Corrector: '{ocr_text}' → '{text}' ({len(corrections)} fixes)")
        
        return text, final_confidence, corrections
    
    def _analyze_structure(self, text: str) -> Dict[str, int]:
        """
        Analyze plate text structure using REGEX CANDIDATE ENUMERATION.
        
        VN Plate Format (Thông tư 58/2020):
        - Province: 2 digits (position 0-1)
        - Series: 1 or 2 letters (position 2 or 2-3)
        - Serial: 4-6 digits (rest)
        
        Returns:
            Dictionary with boundary positions and confidence score
        """
        # Try multiple structure candidates and score them
        candidates = []
        
        # Pattern 1: 2 digits + 1 letter + 4-6 digits (e.g., 29A12345)
        pattern1 = re.match(r'^([0-9]{2})([A-Z])([0-9]{4,6})$', text)
        if pattern1:
            candidates.append({
                'province_start': 0, 'province_end': 2,
                'series_start': 2, 'series_end': 3,
                'series_allow_digit': False,
                'serial_start': 3,
                'score': 1.0  # Highest score for standard format
            })
        
        # Pattern 2: 2 digits + 2 letters + 4-6 digits (e.g., 29AB12345)
        pattern2 = re.match(r'^([0-9]{2})([A-Z]{2})([0-9]{4,6})$', text)
        if pattern2:
            candidates.append({
                'province_start': 0, 'province_end': 2,
                'series_start': 2, 'series_end': 4,
                'series_allow_digit': False,
                'serial_start': 4,
                'score': 0.9  # Slightly lower (less common)
            })
        
        # Pattern 3: 2 digits + letter + digit + 4-6 digits (motorcycle, e.g., 69F112345)
        pattern3 = re.match(r'^([0-9]{2})([A-Z])([0-9])([0-9]{4,6})$', text)
        if pattern3:
            candidates.append({
                'province_start': 0, 'province_end': 2,
                'series_start': 2, 'series_end': 4,
                'series_allow_digit': True,
                'serial_start': 4,
                'score': 0.8  # Motorcycle format
            })
        
        # Fallback: Heuristic parsing if no pattern matches
        if not candidates:
            series_start = 2
            series_end = 3
            series_allow_digit = False
            
            if len(text) >= 4 and text[3].isalpha():
                series_end = 4
            elif len(text) >= 8 and text[3].isdigit():
                series_end = 4
                series_allow_digit = True
            
            return {
                'province_start': 0, 'province_end': 2,
                'series_start': series_start, 'series_end': series_end,
                'series_allow_digit': series_allow_digit,
                'serial_start': series_end,
                'score': 0.5  # Low confidence fallback
            }
        
        # Return best candidate (highest score)
        return max(candidates, key=lambda x: x['score'])
    
    def _fix_province_code(self, text: str, structure: Dict[str, int]) -> Tuple[str, List[str]]:
        """
        Fix province code (positions 0-1) with VALID_PROVINCE_CODES validation.
        
        Rules:
        - Must be exactly 2 digits
        - Must be in VALID_PROVINCE_CODES set (11-99, excluding invalid codes)
        - Uses WEIGHTED confusion matrix to convert letters → digits
        """
        if len(text) < 2:
            return text, []
        
        province_str = text[:2]
        corrections: List[str] = []
        
        # Convert to digits using WEIGHTED confusion matrix
        fixed = ""
        for i, char in enumerate(province_str):
            if char.isdigit():
                fixed += char
            elif char in self._inverse_confusion:
                # Find digit candidates from confusion matrix (sorted by probability)
                candidates = [(c, prob) for c, prob in self._inverse_confusion[char] if c.isdigit()]
                if candidates:
                    # Use highest probability candidate
                    best_char = candidates[0][0]
                    fixed += best_char
                    corrections.append(f"Province[{i}]: {char}→{best_char} (p={candidates[0][1]:.2f})")
                else:
                    fixed += char  # Keep as-is if no digit candidate
            else:
                fixed += char
        
        # Validate against VALID_PROVINCE_CODES
        try:
            province_code = int(fixed)
            if province_code not in self.VALID_PROVINCES:
                # Try to find closest valid province code
                closest = min(self.VALID_PROVINCES, key=lambda x: abs(x - province_code))
                if abs(closest - province_code) <= 5:  # Only fix if close enough
                    fixed = str(closest).zfill(2)
                    corrections.append(f"Province: {province_code}→{closest} (closest valid)")
        except ValueError:
            pass  # Keep as-is if not convertible to int
        
        return fixed + text[2:], corrections
    
    def _fix_series_code(self, text: str, structure: Dict[str, int]) -> Tuple[str, List[str]]:
        """
        Fix series code (positions 2-3 or 2).
        
        Rules:
        - Must be 1 or 2 uppercase letters
        - Valid letters: A-Z except I, O, Q, W (Thông tư 58/2020)
        - Uses confusion matrix to fix invalid letters
        """
        series_start = structure['series_start']
        series_end = structure['series_end']
        
        if len(text) < series_end:
            return text, []
        
        series = text[series_start:series_end]
        corrections: List[str] = []
        
        fixed_series = ""
        series_allow_digit = structure.get('series_allow_digit', False)

        for i, char in enumerate(series):
            if char in self.VALID_SERIES:
                # Already valid - keep it
                fixed_series += char
            elif char.isdigit() and series_allow_digit:
                # Allow digit as 2nd char of series for motorcycle plates
                fixed_series += char
            elif char.isalpha():
                # Invalid letter - try to fix
                if char in self._inverse_confusion:
                    candidates = [c for c in self._inverse_confusion[char] 
                                  if c in self.VALID_SERIES]
                    if candidates:
                        fixed_series += candidates[0]
                        corrections.append(f"Series[{i}]: {char}→{candidates[0]} (invalid→valid)")
                    else:
                        fixed_series += char  # No valid candidate, keep as-is
                else:
                    fixed_series += char
            else:
                # Digit in series position - OCR likely confused it with a letter
                if char in self._inverse_confusion:
                    candidates = [c for c in self._inverse_confusion[char] 
                                  if c in self.VALID_SERIES]
                    if candidates:
                        fixed_series += candidates[0]
                        corrections.append(f"Series[{i}]: {char}(digit)→{candidates[0]}(letter)")
                    else:
                        fixed_series += char
                else:
                    fixed_series += char
        
        result = text[:series_start] + fixed_series + text[series_end:]
        return result, corrections
    
    def _fix_serial_number(self, text: str, structure: Dict[str, int]) -> Tuple[str, List[str]]:
        """
        Fix serial number (positions 4+ or 3+) with SAFE short-serial rescue.
        
        Rules:
        - Must be all digits
        - Length: 5 digits preferred (Thông tư 58/2020), 4-6 tolerated
        - Uses WEIGHTED confusion matrix to convert letters → digits
        - Short-serial rescue: ONLY if raw_confidence < 0.7 AND serial == 4 digits
        """
        serial_start = structure['serial_start']
        
        if len(text) <= serial_start:
            return text, []
        
        serial = text[serial_start:]
        corrections: List[str] = []
        
        # Convert letters to digits using WEIGHTED confusion matrix
        fixed_serial = ""
        for i, char in enumerate(serial):
            if char.isdigit():
                fixed_serial += char
            elif char in self._inverse_confusion:
                # Find digit candidates (sorted by probability)
                candidates = [(c, prob) for c, prob in self._inverse_confusion[char] if c.isdigit()]
                if candidates:
                    best_char = candidates[0][0]
                    fixed_serial += best_char
                    corrections.append(f"Serial[{i}]: {char}→{best_char} (p={candidates[0][1]:.2f})")
                else:
                    fixed_serial += char
            else:
                fixed_serial += char
        
        result = text[:serial_start] + fixed_serial
        
        # SAFE Short-serial rescue: ONLY if serial == 4 digits (not 3 or 5)
        # AND structure score < 0.8 (low confidence parse)
        serial_digits = sum(1 for c in fixed_serial if c.isdigit())
        structure_score = structure.get('score', 1.0)
        
        if serial_digits == 4 and structure_score < 0.8:
            series_end = structure['series_end']
            series_start = structure['series_start']
            if series_end > series_start:
                last_series_char = result[series_end - 1]
                if last_series_char in self._inverse_confusion:
                    digit_alts = [(c, prob) for c, prob in self._inverse_confusion[last_series_char] 
                                  if c.isdigit()]
                    if digit_alts and digit_alts[0][1] >= 0.3:  # Only if probability >= 30%
                        alt_digit = digit_alts[0][0]
                        alt_serial = alt_digit + fixed_serial
                        alt_serial_digits = sum(1 for c in alt_serial if c.isdigit())
                        
                        # Only apply if new series is still valid (at least 1 letter)
                        new_series_len = (series_end - 1) - series_start
                        if new_series_len >= 1 and alt_serial_digits == 5:
                            result = result[:series_end - 1] + alt_serial
                            corrections.append(
                                f"Short-serial rescue: series[-1] '{last_series_char}'→'{alt_digit}' "
                                f"(p={digit_alts[0][1]:.2f}, serial 4→5 digits)"
                            )
                            structure['series_end'] = series_end - 1
                            structure['serial_start'] = series_end - 1
                            logger.debug(
                                f"🔧 Short-serial rescue: '{last_series_char}' → '{alt_digit}' (safe mode)"
                            )
        
        return result, corrections
    
    def _fix_pattern_confusions(self, text: str, structure: Dict[str, int]) -> Tuple[str, List[str]]:
        """
        Fix multi-character pattern confusions.
        
        Examples:
        - "RL" → "R1" (when followed by digits)
        - "rn" → "M" (in certain contexts)
        - "cl" → "D" (in certain contexts)
        
        Context-aware: Only applies fixes where pattern doesn't make sense structurally.
        """
        corrections: List[str] = []
        
        series_start = structure['series_start']
        series_end = structure['series_end']
        
        if len(text) < series_end:
            return text, corrections
        
        series = text[series_start:series_end]
        
        # All 2-char series patterns that are OCR confusions for 'R1'
        # (e.g., OCR confuses '1' after 'R' with 'L', 'N', '!', 'i', 'I', etc.)
        SERIES_SUBSTITUTIONS = {
            'RL': 'R1', 'Rl': 'R1', 'rL': 'R1', 'rl': 'R1',
            'RN': 'R1', 'R!': 'R1', 'Ri': 'R1', 'RI': 'R1',
        }
        if series in SERIES_SUBSTITUTIONS and series_end - series_start == 2:
            remaining = text[series_end:]
            if remaining and all(c.isdigit() for c in remaining):
                fixed = SERIES_SUBSTITUTIONS[series]
                text = text[:series_start] + fixed + remaining
                corrections.append(f"Pattern: {series}→{fixed} (confused char in serial position)")
                structure['series_end'] = series_start + 1
                structure['serial_start'] = series_start + 1
        
        return text, corrections
    
    def _validate_structure(self, text: str) -> bool:
        """
        Validate final plate text against VN plate format regulations.
        
        Valid formats (Thông tư 58/2020):
        - 2 digits + 1 letter + 4-6 digits (e.g., 65A12345)
        - 2 digits + 2 letters + 4-6 digits (e.g., 30AB12345)
        - 2 digits + letter + digit + 4-6 digits (motorcycle, e.g., 69F112345)
        
        Returns:
            True if text matches a valid VN plate pattern
        """
        if len(text) < 7 or len(text) > 10:
            return False
        
        # OPTIMIZED: Single combined regex pattern (faster than 3 separate patterns)
        # Pattern breakdown:
        # ^[1-9][0-9]           - Province code (11-99)
        # [A-Z]{1,2}            - Series (1 or 2 letters)
        # [0-9]?                - Optional digit for motorcycle series (e.g., F1)
        # [0-9]{4,6}$           - Serial number (4-6 digits)
        pattern = rf'^[1-9][0-9][A-Z]{{1,2}}[0-9]?[0-9]{{{SERIAL_DIGIT_MIN},{SERIAL_DIGIT_MAX}}}$'
        
        return bool(re.match(pattern, text))


# Global singleton instance
_corrector_instance: VNPlateCorrector = None
_corrector_lock = __import__('threading').Lock()


def get_corrector() -> VNPlateCorrector:
    """
    Get global VNPlateCorrector instance (singleton pattern).
    
    Returns:
        Singleton corrector instance
    """
    global _corrector_instance
    if _corrector_instance is None:
        with _corrector_lock:
            if _corrector_instance is None:
                _corrector_instance = VNPlateCorrector()
    return _corrector_instance
