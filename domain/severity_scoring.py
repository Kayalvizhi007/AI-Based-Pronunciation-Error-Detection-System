"""
Domain Layer – Severity Scoring
---------------------------------
Assigns a severity level to each PronunciationError based on phoneme
confidence and optional acoustic distance heuristics.
Pure logic – no I/O.
"""

from enum import Enum
from typing import List
from domain.error_detection import PronunciationError, ErrorType


class Severity(str, Enum):
    MINOR    = "minor"
    MODERATE = "moderate"
    SEVERE   = "severe"


# Phoneme pairs that are acoustically very similar → downgrade severity
SIMILAR_PAIRS = {
    frozenset({"P", "B"}),
    frozenset({"T", "D"}),
    frozenset({"K", "G"}),
    frozenset({"F", "V"}),
    frozenset({"S", "Z"}),
    frozenset({"SH", "ZH"}),
    frozenset({"TH", "DH"}),
    frozenset({"CH", "JH"}),
    frozenset({"M", "N"}),
    frozenset({"L", "R"}),
}


def _is_similar_pair(a: str, b: str) -> bool:
    return frozenset({a, b}) in SIMILAR_PAIRS


def score_severity(error: PronunciationError, confidence: float = 0.5) -> Severity:
    """
    Compute a Severity for a single PronunciationError.

    confidence: phoneme-level confidence from the aligner (0.0–1.0).
                Lower confidence → higher severity.

    Rules
    -----
    • Insertions are generally minor (extra sound, less disruptive).
    • Deletions are moderate by default; severe if confidence is very low.
    • Substitutions depend on acoustic similarity and confidence.
    """
    raw = 1.0 - confidence   # 0 = perfect, 1 = very wrong

    if error.error_type == ErrorType.INSERTION:
        return Severity.MINOR

    if error.error_type == ErrorType.DELETION:
        if raw >= 0.7:
            return Severity.SEVERE
        return Severity.MODERATE

    # SUBSTITUTION
    exp = error.expected_phoneme or ""
    det = error.detected_phoneme or ""
    acoustically_close = _is_similar_pair(exp, det)

    if acoustically_close:
        return Severity.MINOR if raw < 0.5 else Severity.MODERATE

    if raw >= 0.7:
        return Severity.SEVERE
    if raw >= 0.4:
        return Severity.MODERATE
    return Severity.MINOR


def annotate_errors(
    errors: List[PronunciationError],
    confidences: List[float],
) -> List[dict]:
    """
    Combine errors with severity scores into dicts ready for the LLM or UI.

    confidences: per-error confidence values in the same order as errors.
    If fewer confidences than errors, 0.5 is used as default.
    """
    annotated = []
    for i, err in enumerate(errors):
        conf = confidences[i] if i < len(confidences) else 0.5
        severity = score_severity(err, confidence=conf)
        d = err.to_dict()
        d["severity"] = severity.value
        d["confidence"] = round(conf, 3)
        annotated.append(d)
    return annotated
