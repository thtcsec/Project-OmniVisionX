"""
PHASE 1: Vietnamese License Plate Validator & Auto-Corrector
=============================================================
Constraint-based grammar validation for VN plates.
- Province code validation (11-99)
- Position-based character rules
- Context-aware OCR error correction
- Confidence penalty for corrections

Vietnamese Plate Format:
- Standard: XX-Y-NNNNN (province + series + number)
- Examples: 29A-123.45, 51G-999.99, 30E-12345
"""

import re
import logging
import os
from typing import Tuple, Optional, Dict, List
from collections import OrderedDict
from dataclasses import dataclass
from app.services.plate.plate_constants import (
    VALID_PROVINCE_CODES, VALID_SERIES_LETTERS as _SHARED_SERIES,
    DIGIT_CONFUSION_GRAPH,
)
from app.services.plate.plate_utils import normalize_plate_basic

logger = logging.getLogger(__name__)


# Series letters — imported from shared constants (includes J, W)
VALID_SERIES_LETTERS = _SHARED_SERIES

# Province digit confusion graph — imported from shared constants
# Used for intelligent province code correction instead of naive ±1
PROVINCE_DIGIT_ALTS = {
    k: [c for c in v if c.isdigit()]
    for k, v in DIGIT_CONFUSION_GRAPH.items()
    if any(c.isdigit() for c in v)
}

# Common OCR confusion patterns (context-aware)
# Format: (pattern, replacement, position_context)
OCR_CORRECTIONS = {
    # Multi-char corrections (context-aware)
    'rn': 'M',   # rn often misread as M
    'vv': 'W',   # vv often misread as W
    'cl': 'D',   # cl often misread as D
    'cI': 'D',   # cI often misread as D
    'll': 'H',   # ll can be H in some fonts
    'lI': 'H',   # lI can be H
    'Vv': 'W',   # Vv -> W
    'VV': 'W',   # VV -> W
}

# Single char corrections by position type
# IMPORTANT: Do NOT include T→7 here — it destroys valid series letters
# T→7 is handled separately in serial-position-only corrections below
DIGIT_CORRECTIONS = {
    'O': '0', 'o': '0',
    'D': '0',             # D↔0 common IR confusion
    'Q': '0',             # Q↔0
    'I': '1', 'l': '1', 'L': '1', 'i': '1',
    'Z': '2', 'z': '2',
    'S': '5', 's': '5',
    'B': '8', 'b': '8',
    'G': '6', 'g': '6',
}

# Letters that often get confused as digits (used for shifted-prefix rescue)
SHIFT_CONFUSABLE = {k.upper() for k in DIGIT_CORRECTIONS.keys() if k.isalpha()}

# Fast grammar gate for candidate filtering (supports optional series digit)
PLATE_REGEX = re.compile(r'^[1-9][0-9][A-Z]{1,2}[0-9]?[0-9]{4,6}$')

# Scoring weights (tunable via env)
SCORE_GRAMMAR_MISMATCH = float(os.getenv("OCR_SCORE_GRAMMAR_MISMATCH", "-0.2"))
SCORE_PROVINCE_INVALID = float(os.getenv("OCR_SCORE_PROVINCE_INVALID", "-0.5"))
SCORE_SERIES_INVALID = float(os.getenv("OCR_SCORE_SERIES_INVALID", "-0.3"))
SCORE_SERIES_INVALID_EXTRA = float(os.getenv("OCR_SCORE_SERIES_INVALID_EXTRA", "-0.1"))
SCORE_SERIAL_NON_DIGIT = float(os.getenv("OCR_SCORE_SERIAL_NON_DIGIT", "-0.4"))
SCORE_LENGTH_OFF = float(os.getenv("OCR_SCORE_LENGTH_OFF", "-0.2"))

# Candidate search params (tunable via env)
CANDIDATE_MAX_EDITS = int(os.getenv("OCR_CANDIDATE_MAX_EDITS", "2"))
CANDIDATE_BEAM_WIDTH = int(os.getenv("OCR_CANDIDATE_BEAM_WIDTH", "6"))

# Additional digit corrections ONLY for serial positions (pos 4+)
# These are excluded from global DIGIT_CORRECTIONS to protect series letters
SERIAL_ONLY_DIGIT_CORRECTIONS = {
    'T': '7',    # T↔7 very common confusion in IR/noisy images
    'J': '7',    # J↔7 similar shape
    'Y': '7',    # Y↔7 in some fonts
    'A': '4',    # A↔4 in serial positions
    'V': '7',    # V↔7 in noisy images
}

LETTER_CORRECTIONS = {
    '0': 'O',
    '1': 'I',
    '8': 'B',
    '6': 'G',
    '5': 'S',
    '4': 'A',             # 4↔A in series position
}

# Confusion candidates for smart correction (single-edit candidates)
CONFUSION_CANDIDATES = {
    **{k: [v] for k, v in DIGIT_CORRECTIONS.items()},
    **{k: [v] for k, v in LETTER_CORRECTIONS.items()},
    **{k: [v] for k, v in SERIAL_ONLY_DIGIT_CORRECTIONS.items()},
}


def _score_plate_candidate(text: str) -> Tuple[float, List[str]]:
    """Score a candidate plate using structure and province validity."""
    errors: List[str] = []
    clean = re.sub(r'[^A-Z0-9]', '', text.upper())
    if len(clean) < 7:
        return 0.0, ["Too short"]

    score = 1.0

    if not PLATE_REGEX.match(clean):
        errors.append("Grammar mismatch")
        score += SCORE_GRAMMAR_MISMATCH

    # Province validity
    is_valid_prov, _ = validate_province_code(clean[:2])
    if not is_valid_prov:
        errors.append("Invalid province")
        score += SCORE_PROVINCE_INVALID

    # Series letters (allow letter+digit for motorcycle plates)
    invalid_series = False
    invalid_series2 = False
    if len(clean) >= 3:
        if clean[2] not in VALID_SERIES_LETTERS:
            invalid_series = True
    if len(clean) >= 4:
        if clean[3].isalpha() and clean[3] not in VALID_SERIES_LETTERS:
            invalid_series2 = True

    if invalid_series or invalid_series2:
        errors.append("Invalid series")
        score += SCORE_SERIES_INVALID
        if invalid_series and invalid_series2:
            score += SCORE_SERIES_INVALID_EXTRA

    # Serial digits
    series_len = 1
    if len(clean) >= 4 and clean[3].isalpha():
        series_len = 2
    elif len(clean) >= 9 and clean[3].isdigit():
        series_len = 2
    serial_start = 2 + series_len
    serial = clean[serial_start:]
    if serial and not serial.isdigit():
        errors.append("Serial non-digit")
        score += SCORE_SERIAL_NON_DIGIT

    if len(clean) not in (8, 9):
        errors.append("Length off")
        score += SCORE_LENGTH_OFF

    return max(0.0, min(1.0, score)), errors


def _generate_candidates(text: str, max_edits: int = 1) -> List[str]:
    """Generate candidates with position-aware substitutions (limited edits)."""
    clean = re.sub(r'[^A-Z0-9]', '', text.upper())
    if max_edits <= 0:
        return [clean] if clean else []
    if len(clean) < 7:
        return [clean] if clean else []

    # Determine expected series length
    series_len = 1
    if len(clean) >= 4 and clean[3].isalpha():
        series_len = 2
    elif len(clean) >= 9 and clean[3].isdigit():
        series_len = 2
    series_end = 2 + series_len
    serial_start = series_end

    serial_corrections = {**DIGIT_CORRECTIONS, **SERIAL_ONLY_DIGIT_CORRECTIONS}
    candidates: set[str] = set()

    for i, ch in enumerate(clean):
        if i < 2:
            alt = DIGIT_CORRECTIONS.get(ch)
            if alt:
                cand = clean[:i] + alt + clean[i + 1:]
                if PLATE_REGEX.match(cand):
                    candidates.add(cand)
        elif i < series_end:
            # Series positions: digit → letter corrections only
            alt = LETTER_CORRECTIONS.get(ch)
            if alt and alt in VALID_SERIES_LETTERS:
                cand = clean[:i] + alt + clean[i + 1:]
                if PLATE_REGEX.match(cand):
                    candidates.add(cand)
            # If pos3 is actually a series digit (L1), allow letter → digit fix
            if i == 3 and series_len == 2:
                alt_digit = DIGIT_CORRECTIONS.get(ch)
                if alt_digit:
                    cand = clean[:i] + alt_digit + clean[i + 1:]
                    if PLATE_REGEX.match(cand):
                        candidates.add(cand)
        else:
            alt = serial_corrections.get(ch)
            if alt:
                cand = clean[:i] + alt + clean[i + 1:]
                if PLATE_REGEX.match(cand):
                    candidates.add(cand)

    # Special series-pair collapse (common OCR split: "11" -> "H")
    if len(clean) >= 4 and clean[2:4] == "11":
        cand = clean[:2] + "H" + clean[4:]
        if PLATE_REGEX.match(cand):
            candidates.add(cand)

    return list(candidates)


def _beam_search_candidates(text: str, max_edits: int, beam_width: int) -> List[str]:
    """
    Beam search over position-aware single-edit candidates.
    Keeps top-N candidates per depth to avoid combinatorial explosion.
    """
    clean = re.sub(r'[^A-Z0-9]', '', text.upper())
    max_edits = max(0, int(max_edits))
    beam_width = max(1, int(beam_width))
    if max_edits <= 0 or not clean:
        return []

    visited = {clean}
    frontier: List[Tuple[str, float]] = [(clean, _score_plate_candidate(clean)[0])]
    results: set[str] = set()

    for _ in range(max_edits):
        next_frontier: List[Tuple[str, float]] = []
        for cand, _score in frontier:
            neighbors = _generate_candidates(cand, max_edits=1)
            for nxt in neighbors:
                if nxt in visited:
                    continue
                visited.add(nxt)
                score, _ = _score_plate_candidate(nxt)
                next_frontier.append((nxt, score))

        if not next_frontier:
            break

        next_frontier.sort(key=lambda x: x[1], reverse=True)
        frontier = next_frontier[:beam_width]
        results.update([c for c, _ in frontier])

    return list(results)


def _generate_shift_candidates(clean: str) -> List[Tuple[str, int]]:
    """
    Generate candidates by removing a possible extra leading character and
    reapplying position-based corrections. This targets OCR shifts like:
    "16SA47208" -> drop leading "1" -> "6SA47208" -> correct -> "65A47208".
    """
    if len(clean) < 9:
        return []
    if not (len(clean) >= 3 and clean[0].isdigit() and clean[1].isdigit()):
        return []
    if clean[2] not in SHIFT_CONFUSABLE:
        return []

    candidates: List[Tuple[str, int]] = []
    for drop_idx in (0, 1):
        trimmed = clean[:drop_idx] + clean[drop_idx + 1:]
        if len(trimmed) < 7:
            continue
        fixed, corrections = correct_plate_by_position(trimmed)
        if fixed and PLATE_REGEX.match(fixed):
            candidates.append((fixed, corrections))
    return candidates


@dataclass
class PlateValidationResult:
    """Result of plate validation"""
    original: str
    corrected: str
    is_valid: bool
    confidence_penalty: float  # 0.0 = no penalty, higher = more corrections made
    province_code: Optional[int]
    series: Optional[str]
    number: Optional[str]
    errors: List[str]


def validate_province_code(code_str: str) -> Tuple[bool, Optional[int]]:
    """
    Validate if province code is valid.
    Returns (is_valid, province_code_int)
    """
    try:
        code = int(code_str)
        if code in VALID_PROVINCE_CODES:
            return True, code
        # Check if close to a valid code (might be OCR error)
        return False, code
    except ValueError:
        return False, None


def apply_context_ocr_fixes(text: str) -> Tuple[str, int]:
    """
    Apply context-aware OCR fixes.
    Returns (fixed_text, num_corrections)
    """
    corrections = 0
    result = text
    
    # Apply multi-char corrections
    for pattern, replacement in OCR_CORRECTIONS.items():
        if pattern in result:
            result = result.replace(pattern, replacement)
            corrections += 1
    
    return result, corrections


def correct_plate_by_position(text: str) -> Tuple[str, int]:
    """
    Apply position-based corrections.
    Vietnamese plate: DD-LL-DDDDD (D=digit, L=letter)
    
    Returns (corrected_text, num_corrections)
    """
    if len(text) < 4:
        return text, 0
    
    corrections = 0
    result = list(text.upper())
    
    # Position 0-1: Must be digits (province code)
    for i in range(min(2, len(result))):
        if result[i] in DIGIT_CORRECTIONS:
            result[i] = DIGIT_CORRECTIONS[result[i]]
            corrections += 1
    
    # Position 2: ALWAYS a series letter — only correct digit→letter when result is VALID
    if len(result) > 2:
        if result[2] in LETTER_CORRECTIONS:
            candidate = LETTER_CORRECTIONS[result[2]]
            if candidate in VALID_SERIES_LETTERS:
                result[2] = candidate
                corrections += 1
            else:
                # '1'→'I' and '0'→'O' are NOT valid series; try secondary mappings
                _SERIES_DIGIT_SECONDARY = {'1': 'T', '0': 'D'}
                alt = _SERIES_DIGIT_SECONDARY.get(result[2])
                if alt and alt in VALID_SERIES_LETTERS:
                    result[2] = alt
                    corrections += 1
        elif result[2].isdigit() and result[2] not in LETTER_CORRECTIONS:
            pass
    
    # Position 3: context-dependent (2-letter series OR first serial digit)
    if len(result) > 3:
        # Count how many digits follow position 4
        tail = ''.join(result[4:]) if len(result) > 4 else ''
        tail_digits = sum(1 for c in tail if c.isdigit())
        total_len = len(result)
        
        # Heuristic: if total length >= 9 and pos3 is a letter → 2-letter series
        # If total length == 8 and pos3 looks like digit → single series + 5-digit serial
        # If pos3 is already a valid series letter, keep it
        if result[3] in VALID_SERIES_LETTERS:
            # Already a valid series letter — keep as-is
            pass
        elif result[3].isalpha() and result[3] not in VALID_SERIES_LETTERS:
            if result[3] in LETTER_CORRECTIONS:
                candidate = LETTER_CORRECTIONS[result[3]]
                if candidate in VALID_SERIES_LETTERS:
                    result[3] = candidate
                    corrections += 1
        elif result[3].isdigit():
            # Digit at pos3:
            # - If total len >= 9 → could be 2-letter series (DD-AB-DDDDD) OR letter+digit
            #   series (DD-A1-DDDDD); both are 9 chars, so 9-char is AMBIGUOUS.
            # - total_len >= 10 is unambiguously 2-letter series + 6-digit serial.
            # - Only convert if the resulting letter is in VALID_SERIES_LETTERS.
            #   This filters '1'→'I' and '0'→'O' which are not valid series letters.
            # - '4'→'A' is excluded at total_len == 9: "F4" is a valid series code
            #   (e.g., 69F412345) and '4' has low visual similarity to 'A'.
            if total_len >= 9 and result[3] in LETTER_CORRECTIONS:
                candidate = LETTER_CORRECTIONS[result[3]]
                # Guard 1: must produce a valid series letter
                if candidate in VALID_SERIES_LETTERS:
                    # Guard 2: '4'→'A' only when total_len >= 10 (unambiguous 2-letter series)
                    if result[3] == '4' and total_len < 10:
                        pass  # Keep as series digit (e.g., F4 in 69F412345)
                    else:
                        result[3] = candidate
                        corrections += 1
            # Otherwise keep digit as-is (serial number or valid series digit)
    
    # Determine where serial digits start
    series_len = 1
    if len(result) > 3:
        if result[3].isalpha():
            series_len = 2
        elif result[3].isdigit() and len(result) >= 9:
            series_len = 2
    serial_start = 2 + series_len
    
    # Remaining positions (serial_start+): Must be digits
    # Apply BOTH global DIGIT_CORRECTIONS and SERIAL_ONLY_DIGIT_CORRECTIONS
    serial_corrections = {**DIGIT_CORRECTIONS, **SERIAL_ONLY_DIGIT_CORRECTIONS}
    for i in range(serial_start, len(result)):
        if result[i] in serial_corrections:
            result[i] = serial_corrections[result[i]]
            corrections += 1
    
    return ''.join(result), corrections


def validate_and_correct_plate(
    plate_text: str,
    original_confidence: float = 1.0
) -> PlateValidationResult:
    """
    PHASE 1: Main validation and correction function.
    
    Args:
        plate_text: Raw OCR output
        original_confidence: Original OCR confidence
    
    Returns:
        PlateValidationResult with corrected plate and adjusted confidence
    """
    errors = []
    
    # Step 1: Apply context-aware OCR fixes on RAW text (before normalize)
    # because OCR_CORRECTIONS contains lowercase patterns like 'rn'→'M', 'cl'→'D'
    # that would never match after .upper() normalization
    pre_fixed, ctx_corrections = apply_context_ocr_fixes(plate_text)

    # Clean input (uppercase + strip non-alphanumeric)
    clean = normalize_plate_basic(pre_fixed)

    if len(clean) < 6:
        return PlateValidationResult(
            original=plate_text,
            corrected=clean,
            is_valid=False,
            confidence_penalty=0.5,
            province_code=None,
            series=None,
            number=None,
            errors=["Plate too short (< 6 chars)"]
        )
    
    # Step 2: Apply position-based corrections (base candidate)
    fixed, pos_corrections = correct_plate_by_position(clean)
    
    total_corrections = ctx_corrections + pos_corrections

    # Step 2.5: Smart candidate search (limit 1 edit) to avoid hard swaps
    candidates = _beam_search_candidates(
        fixed,
        max_edits=CANDIDATE_MAX_EDITS,
        beam_width=CANDIDATE_BEAM_WIDTH,
    )
    best_candidate = fixed
    best_score, _ = _score_plate_candidate(fixed)
    for cand in candidates:
        score, _ = _score_plate_candidate(cand)
        if score > best_score:
            best_score = score
            best_candidate = cand
    if best_candidate != fixed:
        fixed = best_candidate
        total_corrections += 1

    # Step 2.6: Shifted-prefix rescue (extra leading char causes series shift)
    # Only run when OCR confidence is not high OR the current score is weak.
    if original_confidence <= 0.90 or best_score < 0.85:
        shift_candidates = _generate_shift_candidates(clean)
        if shift_candidates:
            current_series_two = (
                len(fixed) >= 4 and fixed[2].isalpha() and fixed[3].isalpha()
            )
            best_shift: Optional[Tuple[str, int]] = None
            best_shift_score = best_score
            for cand, cand_corr in shift_candidates:
                if len(cand) < 4:
                    continue
                # Prefer candidates that become 1-letter series (DD-A-DDDDD)
                if not (cand[2].isalpha() and cand[3].isdigit()):
                    continue
                if not (cand[:2].isdigit() and int(cand[:2]) in VALID_PROVINCE_CODES):
                    continue
                score, _ = _score_plate_candidate(cand)
                if score > best_shift_score or (score == best_shift_score and current_series_two):
                    best_shift_score = score
                    best_shift = (cand, cand_corr)
            if best_shift:
                fixed, shift_corr = best_shift
                best_score = best_shift_score
                total_corrections = ctx_corrections + shift_corr + 1
                logger.debug("🔁 Shifted-prefix rescue applied: %s -> %s", clean, fixed)
    
    # Step 3: Validate province code
    province_str = fixed[:2]
    is_valid_province, province_code = validate_province_code(province_str)
    
    if not is_valid_province:
        errors.append(f"Invalid province code: {province_str}")
        
        # Try digit-confusion graph correction (e.g., 60 -> 68)
        # Uses DIGIT_CONFUSION_GRAPH instead of naive abs() ±1
        if province_code is not None and len(province_str) == 2 and province_str.isdigit():
            base_score, _ = _score_plate_candidate(fixed)
            candidates = set()
            for i, ch in enumerate(province_str):
                for alt in PROVINCE_DIGIT_ALTS.get(ch, []):
                    cand = list(province_str)
                    cand[i] = alt
                    candidates.add(''.join(cand))
            best_cand = None
            best_score = 0.0
            for cand in candidates:
                if cand.isdigit() and int(cand) in VALID_PROVINCE_CODES:
                    score, _ = _score_plate_candidate(cand + fixed[2:])
                    if score > best_score:
                        best_score = score
                        best_cand = cand
            if best_cand:
                candidate_text = best_cand + fixed[2:]
                cand_score, _ = _score_plate_candidate(candidate_text)
                improvement = cand_score - base_score
                if cand_score >= base_score and (original_confidence < 0.9 or improvement > 0.2):
                    fixed = candidate_text
                    total_corrections += 1
                    province_code = int(best_cand)
                    is_valid_province = True
    
    # Step 4: Extract series and number
    series = None
    number = None
    
    if len(fixed) >= 3:
        # Find where series ends (1-2 chars after province)
        series_end = 3
        series_digit = False
        if len(fixed) >= 4 and fixed[3].isalpha():
            series_end = 4
        elif len(fixed) >= 9 and fixed[3].isdigit():
            series_end = 4
            series_digit = True
        
        series = fixed[2:series_end]
        number = fixed[series_end:]
        
        # Validate series
        if series:
            if series[0] not in VALID_SERIES_LETTERS:
                errors.append(f"Invalid series letter: {series}")
            if len(series) == 2:
                if series[1].isalpha() and series[1] not in VALID_SERIES_LETTERS:
                    errors.append(f"Invalid series letter: {series}")
                if series[1].isdigit() and not series_digit:
                    errors.append(f"Invalid series digit: {series}")
        
        # Validate number (should be all digits)
        if number and not number.isdigit():
            errors.append(f"Number contains non-digits: {number}")
    
    # Step 5: Calculate confidence penalty
    # Each correction reduces confidence
    penalty_per_correction = 0.05
    confidence_penalty = min(0.3, total_corrections * penalty_per_correction)
    
    # Additional penalty for invalid province
    if not is_valid_province:
        confidence_penalty += 0.1
    
    # Determine overall validity
    is_valid = is_valid_province and len(errors) <= 1 and len(fixed) >= 7
    
    if total_corrections > 0:
        logger.debug(f"🔧 Plate corrected: {plate_text} -> {fixed} ({total_corrections} fixes, penalty={confidence_penalty:.2f})")
    
    return PlateValidationResult(
        original=plate_text,
        corrected=fixed,
        is_valid=is_valid,
        confidence_penalty=confidence_penalty,
        province_code=province_code,
        series=series,
        number=number,
        errors=errors
    )


def get_adjusted_confidence(original_conf: float, validation_result: PlateValidationResult) -> float:
    """
    Get adjusted confidence after validation/correction.
    """
    adjusted = original_conf - validation_result.confidence_penalty
    
    # Boost if plate is fully valid
    if validation_result.is_valid and len(validation_result.errors) == 0:
        adjusted += 0.05
    
    return max(0.0, min(1.0, adjusted))


# ============================================
# P0.6 FIX: TEMPORAL FORGIVENESS
# ============================================
_LRU_MAX_SIZE = 500  # Max track keys in LRU cache


class TemporalValidator:
    """
    P0.6 FIX: Temporal forgiveness for plate validation.
    
    Uses OrderedDict as LRU cache to prevent memory leaks.
    Max size: 500 track keys (auto-evicts oldest on overflow).
    """
    
    def __init__(self, window_size: int = 5, majority_threshold: float = 0.6):
        self.window_size = window_size
        self.majority_threshold = majority_threshold
        
        # LRU cache: OrderedDict auto-evicts oldest entries
        self._history: OrderedDict[str, List[tuple]] = OrderedDict()
        
        import threading
        self._lock = threading.Lock()
    
    def validate_with_forgiveness(
        self,
        track_key: str,
        plate_text: str,
        original_confidence: float = 1.0
    ) -> PlateValidationResult:
        """
        Validate plate with temporal forgiveness.
        If current frame fails but matches majority of previous frames, accept.
        """
        result = validate_and_correct_plate(plate_text, original_confidence)
        
        with self._lock:
            # LRU: move to end if exists, or create new
            if track_key in self._history:
                self._history.move_to_end(track_key)
            else:
                self._history[track_key] = []
            
            history = self._history[track_key]
            history.append((result.corrected, result.is_valid, original_confidence))
            
            # Keep only recent frames per track
            if len(history) > self.window_size:
                history.pop(0)
            
            # LRU eviction: remove oldest tracks when cache exceeds max
            while len(self._history) > _LRU_MAX_SIZE:
                self._history.popitem(last=False)  # Remove oldest (FIFO)
            
            history_snapshot = list(history)
        
        # Analyze outside lock
        if not result.is_valid and len(history_snapshot) >= 2:
            plate_counts: Dict[str, int] = {}
            for plate, is_valid, conf in history_snapshot:
                normalized = normalize_plate_basic(plate)
                plate_counts[normalized] = plate_counts.get(normalized, 0) + 1
            
            if plate_counts:
                best_plate = max(plate_counts, key=plate_counts.get)
                best_count = plate_counts[best_plate]
                
                if best_count >= len(history_snapshot) * self.majority_threshold:
                    for plate, is_valid, conf in reversed(history_snapshot):
                        normalized = normalize_plate_basic(plate)
                        if normalized == best_plate and is_valid:
                            logger.info(
                                f"🔄 Temporal forgiveness: {result.corrected} -> {plate} "
                                f"(majority={best_count}/{len(history_snapshot)})"
                            )
                            result.corrected = plate
                            result.is_valid = True
                            result.errors = []
                            break
                    else:
                        if best_count >= 2:
                            logger.info(
                                f"🔄 Temporal majority override: accepting {best_plate} "
                                f"(count={best_count}/{len(history_snapshot)})"
                            )
                            for plate, _, _ in reversed(history_snapshot):
                                normalized = normalize_plate_basic(plate)
                                if normalized == best_plate:
                                    result.corrected = plate
                                    result.is_valid = True
                                    result.confidence_penalty = 0.1
                                    result.errors = ["Temporal majority override"]
                                    break
        
        return result
    
    def cleanup_stale_tracks(self, max_age_seconds: float = 120.0):
        """LRU handles eviction automatically. This is a no-op kept for API compat."""
        pass


# Singleton
_temporal_validator: Optional[TemporalValidator] = None
_tv_lock = __import__('threading').Lock()


def get_temporal_validator() -> TemporalValidator:
    """Get singleton TemporalValidator instance."""
    global _temporal_validator
    if _temporal_validator is None:
        with _tv_lock:
            if _temporal_validator is None:
                _temporal_validator = TemporalValidator()
    return _temporal_validator
