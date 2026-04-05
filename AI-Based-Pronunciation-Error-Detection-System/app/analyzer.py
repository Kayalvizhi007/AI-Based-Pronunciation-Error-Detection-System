"""
Application Layer – Pronunciation Analyzer
-------------------------------------------
Flow
----
1. transcribe(audio_path)
     → ASR text + word-level timestamps (faster-whisper)
2. correct_transcript(raw)
     → LLM spelling correction
3. analyze(audio_path, sentence)
     → PronunciationReport with per-word phoneme analysis

The word timestamps from step 1 are stored and forwarded to the alignment
service so it can slice audio per-word rather than running on the full sentence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from domain.error_detection import detect_errors
from domain.learning_logic import compute_accuracy
from domain.severity_scoring import annotate_errors
from infrastructure import database
from infrastructure.database import _now
from services import asr_service, llm_service, mfa_service, phoneme_recognition_service
from services.asr_service import WordTimestamp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Report data structures
# ---------------------------------------------------------------------------

@dataclass
class WordReport:
    word:               str
    expected_phonemes:  List[str]
    detected_phonemes:  List[str]
    errors:             List[dict]
    accuracy:           float
    suggestion:         str

    @property
    def score(self) -> int:
        return round(self.accuracy * 100)

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    @property
    def phoneme_display(self) -> str:
        return " ".join(self.expected_phonemes)

    @property
    def detected_display(self) -> str:
        return " ".join(self.detected_phonemes) if self.detected_phonemes else "—"


@dataclass
class PronunciationReport:
    raw_transcript:     str
    sentence:           str
    word_reports:       List[WordReport]  = field(default_factory=list)
    overall_score:      int               = 0
    overall_suggestion: str               = ""
    session_id:         Optional[int]     = None

    def _compute_overall_score(self) -> None:
        if not self.word_reports:
            self.overall_score = 0
            return
        self.overall_score = round(
            sum(w.accuracy for w in self.word_reports)
            / len(self.word_reports) * 100
        )


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class PronunciationAnalyzer:
    """One instance per Streamlit session (stored in st.session_state)."""

    def __init__(self) -> None:
        database.init_db()
        self._raw_transcript: str = ""
        self._word_timestamps: List[WordTimestamp] = []

    # ── Step 1 ──────────────────────────────────────────────────────────

    def transcribe(self, audio_path: Path) -> str:
        """
        Run ASR and store word-level timestamps for later use by analyze().
        Returns the raw transcript text.
        """
        text, timestamps = asr_service.transcribe_with_word_timestamps(audio_path)
        self._raw_transcript  = text
        self._word_timestamps = timestamps
        logger.info(
            "Transcribed: %r  (%d word timestamps)", text, len(timestamps)
        )
        return text

    # ── Step 2 ──────────────────────────────────────────────────────────

    def correct_transcript(self, raw: str) -> str:
        corrected = llm_service.correct_transcript(raw)
        logger.info("Corrected: %r", corrected)
        return corrected

    # ── Step 3 ──────────────────────────────────────────────────────────

    def analyze(
        self,
        audio_path: Path,
        sentence: str,
        user_id: Optional[int] = None,
    ) -> PronunciationReport:
        """Run full phoneme-level analysis and return PronunciationReport."""
        words = sentence.split()
        logger.info("Analyzing: %r (%d words)", sentence, len(words))

        # Alignment – passes word timestamps so per-word slicing can be used
        alignments = mfa_service.align_audio(
            audio_path,
            sentence,
            asr_transcript=self._raw_transcript,
            word_timestamps=self._word_timestamps or None,
        )

        # Per-word analysis
        word_reports: List[WordReport] = []
        for word in words:
            expected  = mfa_service.get_expected_phonemes(word)
            alignment = next(
                (a for a in alignments if a.word.lower() == word.lower()), None
            )
            detected = alignment.phoneme_sequence if alignment else []
            confs    = [p.confidence for p in alignment.phonemes] if alignment else []

            errors    = detect_errors(word, expected, detected)
            annotated = annotate_errors(errors, confs)
            accuracy  = compute_accuracy(expected, detected)

            suggestion = (
                llm_service.generate_explanation(annotated, word)
                if annotated else ""
            )

            logger.info(
                "word='%s'  expected=%s  detected=%s  acc=%.2f  errs=%d",
                word, expected, detected, accuracy, len(errors),
            )

            word_reports.append(WordReport(
                word=word,
                expected_phonemes=expected,
                detected_phonemes=detected,
                errors=annotated,
                accuracy=accuracy,
                suggestion=suggestion,
            ))

        report = PronunciationReport(
            raw_transcript=self._raw_transcript,
            sentence=sentence,
            word_reports=word_reports,
        )
        report._compute_overall_score()

        session_data = {
            "sentence":      sentence,
            "overall_score": report.overall_score,
            "common_errors": _top_errors(word_reports),
            "word_results": [
                {
                    "word":     wr.word,
                    "score":    wr.score,
                    "errors":   wr.errors,
                    "expected": wr.expected_phonemes,
                    "detected": wr.detected_phonemes,
                }
                for wr in word_reports
            ],
        }
        report.overall_suggestion = llm_service.generate_session_summary(session_data)

        # Persist session with score and feedback
        session_id = database.record_full_session(
            user_id=user_id,
            sentence=sentence,
            score=report.overall_score,
            feedback=report.overall_suggestion,
            audio_file_path=str(audio_path) if audio_path else None,
            started_at=_now(),
            ended_at=_now(),
            summary=report.overall_suggestion,
        )

        # Store detailed per-word and per-phoneme scores for analytics
        for wr in word_reports:
            database.save_word_result(
                session_id=session_id,
                word=wr.word,
                attempts=1,
                passed=wr.accuracy >= 0.70,
                best_accuracy=wr.accuracy,
                errors=wr.errors,
            )
            phoneme_score = round(wr.accuracy * 100)
            database.add_phoneme_score(session_id=session_id, phoneme=wr.word, score=phoneme_score)

        # Update daily operational stats
        day_key = datetime.utcnow().date().isoformat()
        database.upsert_daily_stats(user_id=user_id, date=day_key, session_score=report.overall_score, minutes=3)

        report.session_id = session_id

        return report


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _top_errors(word_reports: List[WordReport]) -> List[str]:
    from collections import Counter
    counts: Counter = Counter()
    for wr in word_reports:
        for e in wr.errors:
            ep = e.get("expected_phoneme")
            if ep:
                counts[ep] += 1
    return [ph for ph, _ in counts.most_common(5)]
