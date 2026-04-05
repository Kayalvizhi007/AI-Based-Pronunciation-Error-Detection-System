"""
Domain Layer – Error Detection
--------------------------------
Compares expected vs detected phoneme sequences and returns structured
PronunciationError objects.  Pure logic – no I/O.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
import difflib


class ErrorType(str, Enum):
    SUBSTITUTION = "substitution"
    DELETION     = "deletion"
    INSERTION    = "insertion"


@dataclass
class PronunciationError:
    """Structured error for a single phoneme mismatch."""
    word: str
    expected_phoneme: Optional[str]   # None for insertion
    detected_phoneme: Optional[str]   # None for deletion
    error_type: ErrorType
    position: int                     # index in expected sequence

    def to_dict(self) -> dict:
        return {
            "word":             self.word,
            "expected_phoneme": self.expected_phoneme,
            "detected_phoneme": self.detected_phoneme,
            "error_type":       self.error_type.value,
            "position":         self.position,
        }


def detect_errors(
    word: str,
    expected: List[str],
    detected: List[str],
) -> List[PronunciationError]:
    """
    Use sequence diffing to find substitutions, deletions, and insertions
    between the expected and detected phoneme sequences.

    Returns a (possibly empty) list of PronunciationError objects.
    """
    errors: List[PronunciationError] = []

    matcher = difflib.SequenceMatcher(None, expected, detected, autojunk=False)
    exp_pos = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            exp_pos += (i2 - i1)
            continue

        if tag == "replace":
            # Align pairs; remainder become deletions or insertions
            exp_chunk = expected[i1:i2]
            det_chunk = detected[j1:j2]
            for k in range(max(len(exp_chunk), len(det_chunk))):
                exp_ph = exp_chunk[k] if k < len(exp_chunk) else None
                det_ph = det_chunk[k] if k < len(det_chunk) else None

                if exp_ph and det_ph:
                    errors.append(PronunciationError(
                        word=word,
                        expected_phoneme=exp_ph,
                        detected_phoneme=det_ph,
                        error_type=ErrorType.SUBSTITUTION,
                        position=i1 + k,
                    ))
                elif exp_ph:
                    errors.append(PronunciationError(
                        word=word,
                        expected_phoneme=exp_ph,
                        detected_phoneme=None,
                        error_type=ErrorType.DELETION,
                        position=i1 + k,
                    ))
                else:
                    errors.append(PronunciationError(
                        word=word,
                        expected_phoneme=None,
                        detected_phoneme=det_ph,
                        error_type=ErrorType.INSERTION,
                        position=i1 + k,
                    ))

        elif tag == "delete":
            for k, ph in enumerate(expected[i1:i2]):
                errors.append(PronunciationError(
                    word=word,
                    expected_phoneme=ph,
                    detected_phoneme=None,
                    error_type=ErrorType.DELETION,
                    position=i1 + k,
                ))

        elif tag == "insert":
            for k, ph in enumerate(detected[j1:j2]):
                errors.append(PronunciationError(
                    word=word,
                    expected_phoneme=None,
                    detected_phoneme=ph,
                    error_type=ErrorType.INSERTION,
                    position=i1,
                ))

    return errors


def has_errors(errors: List[PronunciationError]) -> bool:
    return len(errors) > 0
