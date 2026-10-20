"""
Text Quality Evaluation and Assessment Module.
Evaluates extracted or OCR-processed text for cleanliness, Korean completeness,
repetition, and potential corruption.
"""
import re
from typing import Dict, Any, Optional, List
from config import config

# Regex patterns for linguistic character evaluation
HANGUL_REGEX = re.compile(r"[\uac00-\ud7a3\u1100-\u11ff\u3130-\u318f]")
ALNUM_REGEX = re.compile(r"[a-zA-Z0-9]")
WHITESPACE_REGEX = re.compile(r"\s")
SPECIAL_CHAR_REGEX = re.compile(r"[^\w\s\uac00-\ud7a3\u1100-\u11ff\u3130-\u318f]")
REPEATED_CHAR_REGEX = re.compile(r"(.)\1{3,}")


def evaluate_text_quality(
    text: str,
    ocr_confidence: Optional[float] = None,
    is_korean_expected: bool = True
) -> Dict[str, Any]:
    """
    Evaluates text quality and determines if reprocessing/retry is required.

    Returns:
    {
        "char_count": int,
        "hangul_ratio": float,
        "alnum_ratio": float,
        "special_char_ratio": float,
        "repetition_ratio": float,
        "whitespace_ratio": float,
        "avg_line_length": float,
        "quality_score": float, # 0.0 to 1.0
        "needs_retry": bool,
        "reasons": List[str]
    }
    """
    text_clean = text or ""
    total_len = len(text_clean)

    if total_len == 0:
        return {
            "char_count": 0,
            "hangul_ratio": 0.0,
            "alnum_ratio": 0.0,
            "special_char_ratio": 0.0,
            "repetition_ratio": 0.0,
            "whitespace_ratio": 0.0,
            "avg_line_length": 0.0,
            "quality_score": 0.0,
            "needs_retry": True,
            "reasons": ["Text is empty"]
        }

    hangul_count = len(HANGUL_REGEX.findall(text_clean))
    alnum_count = len(ALNUM_REGEX.findall(text_clean))
    special_count = len(SPECIAL_CHAR_REGEX.findall(text_clean))
    ws_count = len(WHITESPACE_REGEX.findall(text_clean))

    hangul_ratio = round(hangul_count / total_len, 4)
    alnum_ratio = round(alnum_count / total_len, 4)
    special_char_ratio = round(special_count / total_len, 4)
    whitespace_ratio = round(ws_count / total_len, 4)

    # Repetition ratio: count characters in repetitive sequences (e.g. "ㅋㅋㅋㅋ", "......", "????")
    repeated_chars = sum(len(m.group(0)) for m in REPEATED_CHAR_REGEX.finditer(text_clean))
    repetition_ratio = round(repeated_chars / total_len, 4)

    # Line statistics
    lines = [l for l in text_clean.splitlines() if l.strip()]
    avg_line_length = round(total_len / max(1, len(lines)), 2)

    # Penalty scoring
    score = 1.0
    reasons: List[str] = []

    # 1. Extreme shortness penalty
    if total_len < config.min_text_chars_per_page:
        score -= 0.3
        reasons.append(f"Too short ({total_len} chars < {config.min_text_chars_per_page})")

    # 2. Excessive special characters penalty (garbled OCR / noise)
    if special_char_ratio > 0.25:
        score -= min(0.4, (special_char_ratio - 0.25) * 1.5)
        reasons.append(f"High special characters ratio ({special_char_ratio:.2f})")

    # 3. Excessive repetition penalty
    if repetition_ratio > 0.15:
        score -= min(0.3, repetition_ratio * 1.5)
        reasons.append(f"High repetition ratio ({repetition_ratio:.2f})")

    # 4. Korean expectation check
    if is_korean_expected and (hangul_count == 0 and alnum_ratio < 0.4 and total_len > 50):
        score -= 0.4
        reasons.append("Expected Korean text but found none")

    # 5. Low OCR Confidence
    if ocr_confidence is not None:
        if ocr_confidence < config.ocr_min_confidence:
            score -= 0.3
            reasons.append(f"Low OCR confidence ({ocr_confidence:.1f} < {config.ocr_min_confidence})")
        else:
            # Boost if OCR confidence was strong
            score = min(1.0, score + 0.1 * (ocr_confidence / 100.0))

    final_score = round(max(0.0, min(1.0, score)), 4)
    needs_retry = final_score < 0.5 or (ocr_confidence is not None and ocr_confidence < config.ocr_min_confidence)

    return {
        "char_count": total_len,
        "hangul_ratio": hangul_ratio,
        "alnum_ratio": alnum_ratio,
        "special_char_ratio": special_char_ratio,
        "repetition_ratio": repetition_ratio,
        "whitespace_ratio": whitespace_ratio,
        "avg_line_length": avg_line_length,
        "quality_score": final_score,
        "needs_retry": needs_retry,
        "reasons": reasons
    }
