"""
Application Layer - Conversation Controller
-------------------------------------------
Implements the tutoring session as a finite state machine.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import List, Optional

from domain.error_detection import detect_errors
from domain.learning_logic import WordAttempt, WordProgress, compute_accuracy, should_explain
from domain.phoneme_alignment import WordAlignment
from domain.severity_scoring import annotate_errors
from infrastructure import database
from services import asr_service, llm_service, mfa_service
from services import phoneme_recognition_service

logger = logging.getLogger(__name__)


class SessionState(Enum):
    LISTEN_SENTENCE  = auto()
    CONFIRM_SENTENCE = auto()
    PROCESS_WORD     = auto()
    EXPLAIN_ERROR    = auto()
    RETRY_WORD       = auto()
    NEXT_WORD        = auto()
    SESSION_SUMMARY  = auto()


@dataclass
class TutoringSession:
    """All mutable state for one tutoring session."""
    session_id:          Optional[int]       = None
    raw_transcript:      str                 = ""
    corrected_sentence:  str                 = ""
    words:               List[str]           = field(default_factory=list)
    word_index:          int                 = 0
    word_progresses:     List[WordProgress]  = field(default_factory=list)
    alignments:          List[WordAlignment] = field(default_factory=list)
    current_errors:      List[dict]          = field(default_factory=list)
    current_explanation: str                 = ""
    state:               SessionState        = SessionState.LISTEN_SENTENCE
    audio_path:          Optional[Path]      = None
    feedback_message:    str                 = ""
    summary:             str                 = ""

    @property
    def current_word(self) -> Optional[str]:
        return self.words[self.word_index] if self.word_index < len(self.words) else None

    @property
    def current_progress(self) -> Optional[WordProgress]:
        return (
            self.word_progresses[self.word_index]
            if self.word_index < len(self.word_progresses)
            else None
        )

    @property
    def is_complete(self) -> bool:
        return self.state == SessionState.SESSION_SUMMARY


class ConversationController:
    """Drives the tutoring session state machine."""

    def __init__(self, user_id: Optional[int] = None) -> None:
        database.init_db()
        self.session = TutoringSession()
        self.user_id = user_id
        logger.info("Controller initialized (user_id=%s)", user_id)

    def _set_state(self, new_state: SessionState, reason: str = "") -> None:
        old = self.session.state
        self.session.state = new_state
        logger.info("State %s -> %s  (%s)", old.name, new_state.name, reason)

    # ------------------------------------------------------------------
    # Step 1 – record full sentence
    # ------------------------------------------------------------------

    def handle_sentence_audio(self, audio_path: Path) -> None:
        assert self.session.state == SessionState.LISTEN_SENTENCE
        self.session.audio_path     = audio_path
        self.session.raw_transcript = asr_service.transcribe_audio(audio_path)
        logger.info("Raw ASR transcript: %r", self.session.raw_transcript)

        self.session.corrected_sentence = llm_service.correct_transcript(
            self.session.raw_transcript
        )
        logger.info("Corrected sentence: %r", self.session.corrected_sentence)
        self._set_state(SessionState.CONFIRM_SENTENCE, "sentence transcribed")

    # ------------------------------------------------------------------
    # Step 2 – user confirms the sentence
    # ------------------------------------------------------------------

    def confirm_sentence(self, confirmed_sentence: Optional[str] = None) -> None:
        assert self.session.state == SessionState.CONFIRM_SENTENCE

        sentence = confirmed_sentence or self.session.corrected_sentence
        self.session.corrected_sentence = sentence
        self.session.words = sentence.split()
        logger.info("Sentence confirmed: %r (%d words)", sentence, len(self.session.words))

        self.session.word_progresses = [WordProgress(word=w) for w in self.session.words]

        # align_audio now uses wav2vec2 phoneme recognition in its fallback
        # (not ASR text), so it correctly detects mispronunciations in the audio
        self.session.alignments = mfa_service.align_audio(
            self.session.audio_path,
            sentence,
        )
        logger.info("Alignment ready (%d entries)", len(self.session.alignments))

        self.session.session_id = database.start_session(
            sentence=sentence, user_id=self.user_id
        )
        self.session.word_index = 0
        self._set_state(SessionState.PROCESS_WORD, "sentence confirmed")
        self._process_current_word_from_alignment()

    # ------------------------------------------------------------------
    # Step 3 – analyse current word from alignment
    # ------------------------------------------------------------------

    def _process_current_word_from_alignment(self) -> None:
        word = self.session.current_word
        if word is None:
            self._finish_session()
            return

        expected  = mfa_service.get_expected_phonemes(word)
        alignment = self._get_alignment_for_word(word)
        detected  = alignment.phoneme_sequence if alignment else []
        confs     = [p.confidence for p in alignment.phonemes] if alignment else []

        errors    = detect_errors(word, expected, detected)
        annotated = annotate_errors(errors, confs)
        accuracy  = compute_accuracy(expected, detected)

        logger.info(
            "ANALYSIS word='%s'  expected=%s  detected=%s  errors=%d  accuracy=%.2f",
            word, expected, detected, len(errors), accuracy,
        )

        attempt = WordAttempt(
            word=word,
            expected_phonemes=expected,
            detected_phonemes=detected,
            errors=annotated,
            accuracy=accuracy,
        )
        self.session.current_progress.add_attempt(attempt)
        self.session.current_errors = annotated

        if errors and should_explain(annotated):
            self.session.current_explanation = llm_service.generate_explanation(
                annotated, word
            )
            self._set_state(SessionState.EXPLAIN_ERROR, "errors found")
        else:
            self.session.feedback_message = f"✅ '{word}' – well done!"
            self._advance_or_finish()

    # ------------------------------------------------------------------
    # Step 4 – retry: user re-records a single word
    # ------------------------------------------------------------------

    def handle_word_audio(self, audio_path: Path) -> None:
        assert self.session.state in (SessionState.RETRY_WORD, SessionState.EXPLAIN_ERROR)

        word     = self.session.current_word
        expected = mfa_service.get_expected_phonemes(word)

        # Use wav2vec2 phoneme recognition directly on the retry audio.
        # This reads the AUDIO, not the ASR text, so it correctly detects
        # which phonemes were actually produced (e.g. T vs TH).
        detected = phoneme_recognition_service.recognize_phonemes_for_word(
            audio_path=audio_path,
            word=word,
            expected_phonemes=expected,
        )
        # Give every detected phoneme a uniform confidence of 0.8 since
        # we don't have per-phoneme scores from wav2vec2 in this mode
        confs = [0.8] * len(detected)

        errors    = detect_errors(word, expected, detected)
        annotated = annotate_errors(errors, confs)
        accuracy  = compute_accuracy(expected, detected)

        logger.info(
            "RETRY word='%s'  expected=%s  detected=%s  errors=%d  accuracy=%.2f",
            word, expected, detected, len(errors), accuracy,
        )

        attempt = WordAttempt(
            word=word,
            expected_phonemes=expected,
            detected_phonemes=detected,
            errors=annotated,
            accuracy=accuracy,
        )
        progress = self.session.current_progress
        progress.add_attempt(attempt)
        self.session.current_errors = annotated

        if progress.passed:
            self.session.feedback_message = f"✅ '{word}' – great improvement!"
            self._advance_or_finish()
        elif progress.give_up:
            self.session.feedback_message = (
                f"⏭ Moving on from '{word}'. Keep practising this one!"
            )
            self._advance_or_finish()
        else:
            self.session.current_explanation = llm_service.generate_explanation(
                annotated, word
            )
            self._set_state(SessionState.RETRY_WORD, "retry needed")

    # ------------------------------------------------------------------
    # Advance / finish
    # ------------------------------------------------------------------

    def advance_to_next_word(self) -> None:
        self._advance_or_finish()

    def _advance_or_finish(self) -> None:
        progress = self.session.current_progress
        if progress and self.session.session_id:
            last = progress.attempts[-1] if progress.attempts else None
            database.save_word_result(
                session_id=self.session.session_id,
                word=progress.word,
                attempts=progress.attempt_count,
                passed=progress.passed,
                best_accuracy=progress.best_accuracy,
                errors=last.errors if last else [],
            )
            logger.info(
                "Saved word='%s'  attempts=%d  passed=%s  best=%.2f",
                progress.word, progress.attempt_count, progress.passed, progress.best_accuracy,
            )

        self.session.word_index += 1
        if self.session.word_index >= len(self.session.words):
            self._finish_session()
        else:
            self._set_state(SessionState.PROCESS_WORD, "next word")
            self._process_current_word_from_alignment()

    def _finish_session(self) -> None:
        results = database.get_session_results(self.session.session_id)
        session_data = {
            "words_practiced": self.session.words,
            "word_results":    results,
            "common_errors":   self._common_phoneme_errors(),
        }
        summary = llm_service.generate_session_summary(session_data)
        self.session.summary = summary
        database.end_session(session_id=self.session.session_id, summary=summary)
        self._set_state(SessionState.SESSION_SUMMARY, "all words done")

    def _common_phoneme_errors(self) -> list:
        from collections import Counter
        counts = Counter()
        for wp in self.session.word_progresses:
            for attempt in wp.attempts:
                for e in attempt.errors:
                    ep = e.get("expected_phoneme")
                    if ep:
                        counts[ep] += 1
        return [ph for ph, _ in counts.most_common(5)]

    def _get_alignment_for_word(self, word: str) -> Optional[WordAlignment]:
        for a in self.session.alignments:
            if a.word.lower() == word.lower():
                return a
        logger.debug("No alignment found for word='%s'", word)
        return None
