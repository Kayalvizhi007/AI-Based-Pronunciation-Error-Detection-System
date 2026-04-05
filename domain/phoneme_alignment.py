"""
Domain Layer – Phoneme Alignment
---------------------------------
Provides pure-logic utilities for working with phoneme sequences.
No I/O or external calls here – only data transformations.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PhonemeToken:
    """A single phoneme with its time boundaries and confidence score."""
    phoneme: str
    start: float          # seconds
    end: float            # seconds
    confidence: float = 1.0   # 0.0 – 1.0


@dataclass
class WordAlignment:
    """Alignment result for a single word."""
    word: str
    start: float
    end: float
    phonemes: List[PhonemeToken] = field(default_factory=list)

    @property
    def phoneme_sequence(self) -> List[str]:
        """Return just the phoneme labels in order."""
        return [p.phoneme for p in self.phonemes]

    @property
    def average_confidence(self) -> float:
        if not self.phonemes:
            return 0.0
        return sum(p.confidence for p in self.phonemes) / len(self.phonemes)


def build_word_alignment(
    word: str,
    start: float,
    end: float,
    phoneme_data: List[dict],
) -> WordAlignment:
    """
    Construct a WordAlignment from raw MFA/aligner output.

    phoneme_data: list of dicts with keys:
        phoneme (str), start (float), end (float), confidence (float)
    """
    tokens = [
        PhonemeToken(
            phoneme=p["phoneme"],
            start=p["start"],
            end=p["end"],
            confidence=p.get("confidence", 1.0),
        )
        for p in phoneme_data
    ]
    return WordAlignment(word=word, start=start, end=end, phonemes=tokens)
