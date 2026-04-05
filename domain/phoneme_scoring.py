"""
Domain Layer – Phoneme Feature Similarity Scoring
---------------------------------------------------
Replaces binary LCS matching with feature-weighted phoneme similarity.

WHY THIS EXISTS
---------------
The old approach compared phonemes as binary: exact match = 1, anything
else = 0.  This made the scorer far too strict.  Saying "diz" for "these"
(/DH IY Z/ → /D IH Z/) got 1/3 = 33% because only Z matched exactly.

The correct approach, used by industry CAPT systems (e.g. Speechace,
NOVO) and endorsed by phonetics research, is to:
  1. Represent each phoneme as a vector of articulatory features
     (place, manner, voicing, vowel height, vowel backness, tenseness).
  2. Compute similarity as the weighted Jaccard overlap of feature sets.
  3. Use these similarities in a DP alignment (instead of binary 0/1)
     so near-matches contribute partial credit.

TRADEOFF CALIBRATION
--------------------
With this approach, "diz" → "these" scores ~60/100:
  - DH vs D:  same voicing, different place+manner     → ~0.20 similarity
  - IY vs IH: same vowel class, height, backness        → ~0.60 similarity
  - Z  vs Z:  exact match                               → 1.00 similarity
  - Average: 0.60 → 60/100  ✓ (was 33/100 before)

Exact pronunciation = 100/100.
Completely unrelated sounds = ~5-15/100.
Common L2 substitutions (DH→D, IY→IH, TH→S) = 40-65/100.
This matches the "yellow" partial-credit band used by NOVO and Speechace.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List

# ---------------------------------------------------------------------------
# ARPAbet phoneme feature sets
# Each phoneme is described by its articulatory properties.
# ---------------------------------------------------------------------------

_F = frozenset  # shorthand

PHONEME_FEATURES: Dict[str, FrozenSet[str]] = {
    # ── Stops ────────────────────────────────────────────────────────────
    "P":  _F({"stop", "bilabial",      "unvoiced"}),
    "B":  _F({"stop", "bilabial",      "voiced"}),
    "T":  _F({"stop", "alveolar",      "unvoiced"}),
    "D":  _F({"stop", "alveolar",      "voiced"}),
    "K":  _F({"stop", "velar",         "unvoiced"}),
    "G":  _F({"stop", "velar",         "voiced"}),
    # ── Fricatives ───────────────────────────────────────────────────────
    "F":  _F({"fricative", "labiodental", "unvoiced"}),
    "V":  _F({"fricative", "labiodental", "voiced"}),
    "TH": _F({"fricative", "dental",      "unvoiced"}),
    "DH": _F({"fricative", "dental",      "voiced"}),
    "S":  _F({"fricative", "alveolar",    "unvoiced"}),
    "Z":  _F({"fricative", "alveolar",    "voiced"}),
    "SH": _F({"fricative", "postalveolar","unvoiced"}),
    "ZH": _F({"fricative", "postalveolar","voiced"}),
    "HH": _F({"fricative", "glottal",     "unvoiced"}),
    # ── Affricates ───────────────────────────────────────────────────────
    "CH": _F({"affricate", "postalveolar", "unvoiced"}),
    "JH": _F({"affricate", "postalveolar", "voiced"}),
    # ── Nasals ───────────────────────────────────────────────────────────
    "M":  _F({"nasal", "bilabial",  "voiced"}),
    "N":  _F({"nasal", "alveolar",  "voiced"}),
    "NG": _F({"nasal", "velar",     "voiced"}),
    # ── Approximants / Liquids ───────────────────────────────────────────
    "L":  _F({"lateral",     "alveolar",      "voiced"}),
    "R":  _F({"approximant", "postalveolar",  "voiced"}),
    "W":  _F({"approximant", "bilabial",      "voiced", "velar"}),
    "Y":  _F({"approximant", "palatal",       "voiced"}),
    # ── Vowels (high front) ──────────────────────────────────────────────
    "IY": _F({"vowel", "high", "front", "tense"}),
    "IH": _F({"vowel", "high", "front", "lax"}),
    # ── Vowels (mid front) ───────────────────────────────────────────────
    "EY": _F({"vowel", "mid",  "front", "tense", "diphthong"}),
    "EH": _F({"vowel", "mid",  "front", "lax"}),
    "AE": _F({"vowel", "low",  "front", "lax"}),
    # ── Vowels (central) ─────────────────────────────────────────────────
    "AH": _F({"vowel", "mid",  "central", "lax"}),
    "ER": _F({"vowel", "mid",  "central", "rhotic"}),
    # ── Vowels (back) ────────────────────────────────────────────────────
    "AA": _F({"vowel", "low",  "back", "lax"}),
    "AO": _F({"vowel", "mid",  "back", "tense"}),
    "OW": _F({"vowel", "mid",  "back", "tense", "diphthong"}),
    "UH": _F({"vowel", "high", "back", "lax"}),
    "UW": _F({"vowel", "high", "back", "tense"}),
    # ── Diphthongs ───────────────────────────────────────────────────────
    "AW": _F({"vowel", "low",  "front", "diphthong"}),
    "AY": _F({"vowel", "low",  "front", "diphthong"}),
    "OY": _F({"vowel", "mid",  "back",  "diphthong"}),
}

# ---------------------------------------------------------------------------
# Similarity function
# ---------------------------------------------------------------------------

def phoneme_similarity(ph1: str, ph2: str) -> float:
    """
    Feature-weighted Jaccard similarity between two ARPAbet phonemes.

    Returns
    -------
    1.0  – identical phonemes
    0.5-0.9  – same class (vowel/consonant), different realisation
    0.1  – vowel vs consonant (maximally different)
    0.0  – one or both phonemes unknown
    """
    if ph1 == ph2:
        return 1.0

    f1 = PHONEME_FEATURES.get(ph1)
    f2 = PHONEME_FEATURES.get(ph2)

    if f1 is None or f2 is None:
        return 0.0

    # Vowel ↔ consonant substitution: very different, but give tiny credit
    is_vowel1 = "vowel" in f1
    is_vowel2 = "vowel" in f2
    if is_vowel1 != is_vowel2:
        return 0.10

    # Within same class: Jaccard similarity
    intersection = len(f1 & f2)
    union        = len(f1 | f2)
    return intersection / union if union else 0.0


# ---------------------------------------------------------------------------
# Feature-weighted accuracy (replaces binary LCS)
# ---------------------------------------------------------------------------

def compute_lenient_accuracy(expected: List[str], detected: List[str]) -> float:
    """
    Compute pronunciation accuracy with partial credit for near-matches.

    Uses a DP alignment (like edit distance) where the match score between
    two phonemes is their feature-weighted similarity (0.0–1.0) rather than
    a binary 0/1.

    The final score is normalised to [0.0, 1.0] by dividing by
    len(expected), so it degrades gracefully with deletions and insertions.

    Examples
    --------
    "these" (DH IY Z) vs "diz" (D IH Z):
        - DH/D  → similarity ≈ 0.20
        - IY/IH → similarity ≈ 0.60
        - Z/Z   → similarity  = 1.00
        - score = 1.80 / 3 = 0.60  →  60 / 100  ✓

    Perfect match:
        score = 3.0 / 3 = 1.0  → 100 / 100  ✓

    All deletions (nothing detected):
        score = 0.0 / 3 = 0.0  →  0 / 100  ✓
    """
    if not expected:
        return 1.0
    if not detected:
        return 0.0

    m = len(expected)
    n = len(detected)

    # dp[i][j] = best cumulative similarity for expected[:i] vs detected[:j]
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            sim = phoneme_similarity(expected[i - 1], detected[j - 1])
            dp[i][j] = max(
                dp[i - 1][j - 1] + sim,   # align expected[i] with detected[j]
                dp[i - 1][j],              # skip expected phoneme (deletion)
                dp[i][j - 1],              # skip detected phoneme (insertion)
            )

    raw = dp[m][n] / m  # normalise to expected length
    return min(1.0, raw)
