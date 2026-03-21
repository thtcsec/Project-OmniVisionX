"""
2-Line CTC Decoder with Confidence-Aware Beam Search + VN Syntax Gate
=====================================================================
Đọc biển số 2 hàng (biển số mô tô, tải xe)
- 2 heads CTC: one for top line, one for bottom line
- Joint decoding for positional relationship
- Confidence-based filtering
- VN Syntax Gate: beam search candidates validated against Vietnamese
  plate format (Thông tư 58/2020/TT-BCA). Valid candidates are boosted,
  invalid candidates are demoted.

Based on research from DeepSeek/Gemini:
- DeepSeek: "Beam Search Decoding + regex filter → chọn candidate hợp lệ nhất"
- Gemini: "Syntax Checker acts as a final gate, checks against VN plate rules"
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
import re
import logging

logger = logging.getLogger(__name__)


class BeamSearchState:
    """State for beam search decoding"""

    def __init__(self, sequence: List[int], confidence: float, prob: float):
        self.sequence = sequence
        self.confidence = confidence  # Min confidence in sequence
        self.prob = prob  # Log probability

    def __lt__(self, other):
        # Sort by confidence first, then by probability
        if self.confidence != other.confidence:
            return self.confidence > other.confidence  # Higher confidence first
        return self.prob > other.prob  # Higher prob first


class CTCDecoder:
    """Single-line CTC decoder (for top or bottom line)"""

    def __init__(self, vocab: str, blank_idx: int = 0):
        self.vocab = vocab
        self.blank_idx = blank_idx
        # Model output dim must be len(vocab) + 1 to separate blank from chars.
        # Index 0 = blank, indices 1..len(vocab) = actual characters.
        self.vocab_size = len(vocab) + 1

    def greedy_decode(self, outputs: torch.Tensor) -> Tuple[str, float]:
        predictions = torch.argmax(outputs, dim=-1)

        decoded = []
        confidences = []

        prev_idx = self.blank_idx
        for t, pred_idx in enumerate(predictions):
            if pred_idx != self.blank_idx and pred_idx != prev_idx:
                # Chars start at index 1 (index 0 = blank)
                char_idx = pred_idx - 1
                if 0 <= char_idx < len(self.vocab):
                    decoded.append(self.vocab[char_idx])
                    confidences.append(outputs[t, pred_idx].item())
            prev_idx = pred_idx

        text = ''.join(decoded)
        confidence = np.mean(confidences) if confidences else 0.0

        return text, confidence

    def beam_search_decode(self, outputs: torch.Tensor, beam_width: int = 10,
                          min_confidence: float = 0.25) -> List[Tuple[str, float]]:
        T, V = outputs.shape

        states = [BeamSearchState(sequence=[], confidence=1.0, prob=0.0)]

        for t in range(T):
            frame_probs = outputs[t]

            candidates = []

            for state in states:
                for char_idx in range(V):
                    prob_val = frame_probs[char_idx].item()

                    if char_idx != self.blank_idx and prob_val < min_confidence:
                        continue

                    new_seq = state.sequence.copy()
                    new_confidence = state.confidence

                    if char_idx != self.blank_idx:
                        if not new_seq or new_seq[-1] != char_idx:
                            new_seq.append(char_idx)
                            new_confidence = min(new_confidence, prob_val)

                    new_prob = state.prob + np.log(max(prob_val, 1e-8))

                    candidates.append(
                        BeamSearchState(new_seq, new_confidence, new_prob)
                    )

            candidates.sort()
            states = candidates[:beam_width]

        results = []
        for state in states:
            # Chars start at index 1 (index 0 = blank)
            text = ''.join([self.vocab[idx - 1] for idx in state.sequence
                            if 0 <= idx - 1 < len(self.vocab)])
            results.append((text, state.confidence))

        return results


# ============================================
# VN SYNTAX GATE
# ============================================

# Valid Vietnamese province codes (Thông tư 58/2020/TT-BCA)
_VN_PROVINCE_CODES = {
    11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
    30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
    66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,
    84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
}

# VN plate format regexes (covers 1-line and 2-line variants)
_VN_PLATE_PATTERNS = [
    # 2-digit province + 1 letter + 4-6 digits (e.g., 30A12345)
    re.compile(r'^(\d{2})([A-Z])(\d{4,6})$'),
    # 2-digit province + 2 letters + 4-6 digits (e.g., 30AB12345)
    re.compile(r'^(\d{2})([A-Z]{2})(\d{4,6})$'),
    # 2-digit province + 1 letter + 1 digit + 4-5 digits (e.g., 59F112345)
    re.compile(r'^(\d{2})([A-Z])(\d)(\d{4,5})$'),
]

# Top-line only patterns (for 2-line plates)
_VN_TOP_PATTERNS = [
    re.compile(r'^(\d{2})([A-Z]{1,2})$'),          # 29A, 29AB
    re.compile(r'^(\d{2})([A-Z]{1,2})(\d)$'),      # 29A1, 51F1 (motorcycle)
]

# Bottom-line only patterns
_VN_BOTTOM_PATTERNS = [
    re.compile(r'^(\d{4,6})$'),
    re.compile(r'^([A-Z0-9]{4,6})$'),
]


def vn_syntax_score(text: str) -> float:
    """
    VN Syntax Gate: Score a candidate plate string.
    
    Returns multiplier:
    - 1.2 = perfect VN format with valid province code
    - 1.1 = valid VN format but unknown province code
    - 1.0 = can't determine (neutral)
    - 0.7 = clearly invalid format (penalty)
    """
    if not text or len(text) < 5:
        return 0.7
    
    # Check full plate patterns
    for pattern in _VN_PLATE_PATTERNS:
        m = pattern.match(text)
        if m:
            province = int(m.group(1))
            if province in _VN_PROVINCE_CODES:
                return 1.2  # Perfect match
            elif 11 <= province <= 99:
                return 1.1  # Valid format, unknown province
            return 0.8  # Invalid province
    
    # If no pattern matches but length is reasonable, neutral
    if len(text) >= 7 and text[:2].isdigit():
        return 0.9
    
    return 0.7  # Penalty for clearly wrong format


def vn_syntax_score_line(text: str, is_top: bool = True) -> float:
    """
    Score a single line (top or bottom) of a 2-line plate.
    
    Returns multiplier:
    - 1.2 = perfect format with valid province
    - 1.0 = neutral
    - 0.7 = invalid
    """
    if not text:
        return 0.7
    
    if is_top:
        for pattern in _VN_TOP_PATTERNS:
            m = pattern.match(text)
            if m:
                province = int(m.group(1))
                if province in _VN_PROVINCE_CODES:
                    return 1.2
                return 0.9
        return 0.7 if len(text) >= 2 else 0.5
    else:
        for pattern in _VN_BOTTOM_PATTERNS:
            if pattern.match(text):
                return 1.1
        return 0.8


class TwoLineCTCDecoder:
    """
    2-Line CTC Decoder for Vietnamese license plates with VN Syntax Gate.
    - Top line: Province code + Series (e.g., "29A")
    - Bottom line: Sequence number (e.g., "123456")
    - Syntax Gate: Validates beam search candidates against VN plate rules
    """

    def __init__(self, vocab_top: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-",
                 vocab_bottom: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        self.decoder_top = CTCDecoder(vocab_top)
        self.decoder_bottom = CTCDecoder(vocab_bottom)

        # Vietnamese plate validation regex (expanded for 1-2 letter series + digit suffix)
        self.plate_regex_top = re.compile(r'^(\d{2})([A-Z]{1,2})(\d?)$')
        self.plate_regex_bottom = re.compile(r'^([A-Z0-9]{4,6})$')

        # Full province codes from plate_constants.py (Thông tư 58/2020)
        self.province_codes = {str(c) for c in _VN_PROVINCE_CODES}

        self.series_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

    def _validate_top_line(self, text: str) -> bool:
        if not self.plate_regex_top.match(text):
            return False

        province = text[:2]
        return province in self.province_codes

    def _validate_bottom_line(self, text: str) -> bool:
        return bool(self.plate_regex_bottom.match(text))

    def _correct_common_errors(self, text: str, is_top: bool = True) -> str:
        if not text:
            return text

        corrections = {
            'top': {
                'B': '8', '8': 'B',
                'D': '0', '0': 'D',
                'I': '1', '1': 'I',
                'L': '1',
                'S': '5', '5': 'S',
                'O': '0', 'Q': '0',
            },
            'bottom': {
                'O': '0', 'Q': '0',
                'I': '1', 'L': '1',
                'S': '5', 'B': '8', 'Z': '2', 'G': '6', 'D': '0',
            }
        }

        mapping = corrections['top'] if is_top else corrections['bottom']
        validate = self._validate_top_line if is_top else self._validate_bottom_line

        if validate(text):
            return text

        for i, ch in enumerate(text):
            repl = mapping.get(ch)
            if not repl:
                continue
            candidate = text[:i] + repl + text[i + 1:]
            if validate(candidate):
                return candidate

        return text

    def _joint_decode(self, output_top: torch.Tensor, output_bottom: torch.Tensor,
                     beam_width: int = 10) -> List[Tuple[str, str, float]]:
        """
        Joint beam search decoding with VN Syntax Gate.

        Instead of hard-rejecting invalid candidates, we use syntax scoring
        to boost valid VN plates and demote invalid ones. This allows the
        system to produce fallback results even when no candidate perfectly
        matches the expected format.

        Syntax Gate (from DeepSeek/Gemini research):
        - Valid top + valid bottom → 1.2× confidence boost
        - Invalid format → 0.7× confidence penalty
        """
        candidates_top = self.decoder_top.beam_search_decode(output_top, beam_width, min_confidence=0.25)
        candidates_bottom = self.decoder_bottom.beam_search_decode(output_bottom, beam_width, min_confidence=0.25)

        results = []

        for top_text, top_conf in candidates_top:
            # Apply syntax gate scoring instead of hard rejection
            top_syntax = vn_syntax_score_line(top_text, is_top=True)
            
            # Try error correction if syntax score is low
            effective_top = top_text
            effective_top_conf = top_conf * top_syntax
            
            if top_syntax < 1.0:
                corrected = self._correct_common_errors(top_text, is_top=True)
                corrected_syntax = vn_syntax_score_line(corrected, is_top=True)
                if corrected_syntax > top_syntax:
                    effective_top = corrected
                    effective_top_conf = top_conf * corrected_syntax * 0.95  # Small penalty for correction

            for bottom_text, bottom_conf in candidates_bottom:
                # Apply syntax gate to bottom line too
                bottom_syntax = vn_syntax_score_line(bottom_text, is_top=False)
                effective_bottom_conf = bottom_conf * bottom_syntax

                # Joint confidence with syntax gate applied
                joint_conf = min(effective_top_conf, effective_bottom_conf)

                results.append((effective_top, bottom_text, joint_conf))

        results.sort(key=lambda x: x[2], reverse=True)

        return results

    def decode(self, output_top: torch.Tensor, output_bottom: torch.Tensor,
              beam_width: int = 10, use_beam_search: bool = True) -> Dict:
        if use_beam_search:
            candidates = self._joint_decode(output_top, output_bottom, beam_width)
        else:
            top_text, top_conf = self.decoder_top.greedy_decode(output_top)
            bottom_text, bottom_conf = self.decoder_bottom.greedy_decode(output_bottom)
            candidates = [(top_text, bottom_text, min(top_conf, bottom_conf))]

        if not candidates:
            top_text, top_conf = self.decoder_top.greedy_decode(output_top)
            bottom_text, bottom_conf = self.decoder_bottom.greedy_decode(output_bottom)
            candidates = [(top_text, bottom_text, min(top_conf, bottom_conf))]

        top_text, bottom_text, confidence = candidates[0]

        return {
            'plate': f"{top_text}-{bottom_text}",
            'top': top_text,
            'bottom': bottom_text,
            'confidence': float(confidence),
            'candidates': candidates[:3],
        }


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    decoder = TwoLineCTCDecoder()

    T = 15
    vocab_top_size = decoder.decoder_top.vocab_size
    vocab_bottom_size = decoder.decoder_bottom.vocab_size

    output_top = torch.randn(T, vocab_top_size, device=device)
    output_top = F.softmax(output_top, dim=-1)

    output_bottom = torch.randn(T, vocab_bottom_size, device=device)
    output_bottom = F.softmax(output_bottom, dim=-1)

    result = decoder.decode(output_top, output_bottom, beam_width=5)

    print(f"Plate: {result['plate']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Candidates: {result['candidates']}")
