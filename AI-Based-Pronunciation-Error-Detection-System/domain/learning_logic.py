"""
Domain Layer – Learning Logic
------------------------------
Pass/fail thresholds and per-attempt tracking.
Uses feature-weighted phoneme similarity for scoring (not binary LCS).
"""

from dataclasses import dataclass, field
from typing import List
from domain.severity_scoring import Severity
from domain.phoneme_scoring import compute_lenient_accuracy   # ← new lenient scorer


MAX_RETRIES     = 3     # max retry attempts per word
PASS_THRESHOLD  = 0.70  # ≥70% feature-weighted similarity = passed
                        # (was 0.75 binary; 0.70 lenient is roughly equivalent)


@dataclass
class WordAttempt:
    word:              str
    expected_phonemes: List[str]
    detected_phonemes: List[str]
    errors:            List[dict]
    accuracy:          float         # 0.0 – 1.0  (feature-weighted)


@dataclass
class WordProgress:
    word:     str
    attempts: List[WordAttempt] = field(default_factory=list)

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)

    @property
    def passed(self) -> bool:
        if not self.attempts:
            return False
        return self.attempts[-1].accuracy >= PASS_THRESHOLD

    @property
    def give_up(self) -> bool:
        return self.attempt_count >= MAX_RETRIES and not self.passed

    @property
    def best_accuracy(self) -> float:
        if not self.attempts:
            return 0.0
        return max(a.accuracy for a in self.attempts)

    def add_attempt(self, attempt: WordAttempt) -> None:
        self.attempts.append(attempt)


def compute_accuracy(expected: List[str], detected: List[str]) -> float:
    """
    Feature-weighted phoneme accuracy (0.0 – 1.0).
    Delegates to compute_lenient_accuracy for partial credit on near-matches.
    """
    return compute_lenient_accuracy(expected, detected)


def should_explain(errors: List[dict], severity_threshold: str = "minor") -> bool:
    order = [Severity.MINOR, Severity.MODERATE, Severity.SEVERE]
    threshold_idx = order.index(Severity(severity_threshold))
    for e in errors:
        if order.index(Severity(e["severity"])) >= threshold_idx:
            return True
    return False
