"""Feedback generator for pronunciation training.

This module contains heuristic rules to generate actionable improvement
points based on phoneme error patterns. It is intentionally light-weight and
does not require any external LLM.
"""

from __future__ import annotations

from collections import Counter
from typing import List


def word_feedback_points(word: str, errors: List[dict]) -> List[str]:
    """Generate a small set of feedback suggestions for a single word."""
    points: List[str] = []

    if not errors:
        points.append(f"Well done! '{word}' sounded clear and natural.")
        return points

    # Collect errors by type
    subs = [e for e in errors if e.get("error_type") == "substitution"]
    dels = [e for e in errors if e.get("error_type") == "deletion"]
    ins = [e for e in errors if e.get("error_type") == "insertion"]

    if subs:
        # show up to two substitution patterns
        seen = set()
        for e in subs[:2]:
            exp = e.get("expected_phoneme") or "?"
            det = e.get("detected_phoneme") or "?"
            if (exp, det) in seen:
                continue
            seen.add((exp, det))
            points.append(
                f"Try saying /{exp}/ instead of /{det}/ — focus on the mouth position for /{exp}/."
            )

    if dels:
        exp = dels[0].get("expected_phoneme") or "?"
        points.append(
            f"You may be dropping the /{exp}/ sound. Slow down and make sure you finish the word."
        )

    if ins:
        det = ins[0].get("detected_phoneme") or "?"
        points.append(
            f"Avoid adding extra sounds like /{det}/. Try to say the word in one clean attempt."
        )

    # Add some general practice tips
    if len(points) < 4:
        points.append("Repeat the word slowly, exaggerating the mouth shape for each sound.")
    if len(points) < 5:
        points.append("Listen to the correct pronunciation audio and try to imitate the rhythm.")
    if len(points) < 6:
        points.append("Record yourself a few times and compare to the correct version.")

    return points


def session_feedback_points(report) -> List[str]:
    """Generate a set of high-level improvement points for the full session."""
    points: List[str] = []

    # Common error sounds
    all_errors = [(wr.word, e) for wr in report.word_reports for e in wr.errors]
    if not all_errors:
        return [
            "Excellent! All words were pronounced clearly.",
            "Keep practising to maintain this clarity.",
            "Try longer sentences to challenge yourself further.",
            "Record and listen to your voice to build confidence.",
            "Keep focusing on rhythm and stress patterns.",
            "You're doing great — consistency is key!",
        ]

    # Key issues
    by_expected = Counter(
        e.get("expected_phoneme") for _, e in all_errors if e.get("expected_phoneme")
    )
    common = [ph for ph, _ in by_expected.most_common(3)]

    points.append("Slow down your speech to improve phoneme clarity.")
    if common:
        points.append(
            "Focus on producing these sounds correctly: "
            + ", ".join(f"/{ph}/" for ph in common)
            + "."
        )
    points.append("Practise vowel sounds like /UW/ and /IY/ by holding them longer.")
    points.append("Avoid inserting extra sounds mid-word; keep words smooth and continuous.")
    points.append("Listen carefully to the reference audio and repeat each word 3–5 times.")

    # Strengths and improvements points
    strong_words = [wr for wr in report.word_reports if wr.score >= 90]
    weak_words = [wr for wr in report.word_reports if wr.score < 70]
    if strong_words:
        points.append(
            "You pronounced words like "
            + ", ".join(wr.word for wr in strong_words[:3])
            + " clearly — keep using that mouth shape."
        )
    if weak_words and len(points) < 6:
        points.append(
            "Spend extra time on: "
            + ", ".join(wr.word for wr in weak_words[:3])
            + "."
        )

    # Ensure at least 6 points
    while len(points) < 6:
        points.append("Keep practising consistently; even 5 minutes per day makes a big difference.")

    return points
