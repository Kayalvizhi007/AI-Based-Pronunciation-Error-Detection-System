"""
Services Layer – MFA / Phoneme Alignment Service
-------------------------------------------------

Alignment pipeline (tried in order):
  1. MFA (Montreal Forced Aligner) – gold standard, needs conda install
  2. Whisper word timestamps + per-word wav2vec2  ← main fallback
  3. Sentence-level wav2vec2 + proportional split  ← last resort
  4. CMUdict ASR-word comparison                   ← ultimate fallback

Why per-word slicing matters
-----------------------------
Running wav2vec2 on a full sentence and distributing phonemes proportionally
across 40+ words breaks down because:
  - CTC collapses long silence → very few output tokens
  - Proportional distribution starves words at the end
  - Result: first 3-4 words get all phonemes, rest score 0/100

The correct approach (path 2): Whisper gives us word timestamps (start/end
seconds per word), we slice the audio into one clip per word, and run
phoneme recognition on each clip independently.  This gives accurate
per-word phoneme detection regardless of sentence length.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import nltk
from nltk.corpus import cmudict

from domain.phoneme_alignment import WordAlignment, build_word_alignment
from services import phoneme_recognition_service
from services.asr_service import WordTimestamp

logger = logging.getLogger(__name__)

_cmu: Optional[dict] = None

MFA_ACOUSTIC_MODEL = os.getenv("MFA_ACOUSTIC_MODEL", "english_us_arpa")
MFA_DICTIONARY     = os.getenv("MFA_DICTIONARY",     "english_us_arpa")

_PUNCT = re.compile(r"[^\w\s'-]")

SAMPLE_RATE = 16_000
# Minimum clip length in seconds – clips shorter than this are too short for
# wav2vec2 to give reliable results.
MIN_CLIP_SEC = 0.25
# Padding added each side of a word timestamp (seconds) to avoid clipping.
# 0.10 s is enough for a pitch period at 100 Hz and gives the model context
# without blending neighbouring words too much.
CLIP_PAD_SEC = 0.10


# ---------------------------------------------------------------------------
# CMUdict helpers
# ---------------------------------------------------------------------------

def _get_cmudict() -> dict:
    global _cmu
    if _cmu is None:
        try:
            _cmu = cmudict.dict()
        except LookupError:
            nltk.download("cmudict", quiet=True)
            _cmu = cmudict.dict()
    return _cmu


def _clean_word(word: str) -> str:
    return _PUNCT.sub("", word).strip().lower()


def get_expected_phonemes(word: str) -> List[str]:
    """Return CMUdict pronunciation for word, stripping punctuation and stress."""
    clean = _clean_word(word)
    if not clean:
        return []
    entries = _get_cmudict().get(clean)
    if not entries:
        logger.debug("No CMUdict entry for word='%s' (cleaned='%s')", word, clean)
        return []
    return [ph.rstrip("012") for ph in entries[0]]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def align_audio(
    audio_path: str | Path,
    transcript: str,
    asr_transcript: Optional[str] = None,
    word_timestamps: Optional[List[WordTimestamp]] = None,
) -> List[WordAlignment]:
    """
    Align audio against a reference transcript and return per-word phonemes.

    Parameters
    ----------
    audio_path       : path to the recorded audio
    transcript       : the confirmed (corrected) sentence
    asr_transcript   : raw ASR text (used in last-resort fallback)
    word_timestamps  : Whisper word-level timestamps (enables per-word slicing)
    """
    logger.info("align_audio — transcript=%r", transcript)

    # Path 1 – MFA
    try:
        alignments = _mfa_align(audio_path, transcript)
        logger.info("MFA alignment succeeded (%d words)", len(alignments))
        return alignments
    except Exception as exc:
        logger.warning("MFA unavailable (%s)", exc)

    # Path 2 – Whisper timestamps → per-word wav2vec2
    if word_timestamps:
        try:
            alignments = _per_word_slice_align(
                audio_path, transcript, word_timestamps
            )
            logger.info("Per-word slice alignment succeeded (%d words)", len(alignments))
            return alignments
        except Exception as exc:
            logger.warning("Per-word slice alignment failed (%s)", exc)

    # Path 3 – full-sentence wav2vec2 + proportional split (unreliable for long sentences)
    try:
        alignments = _sentence_level_fallback(audio_path, transcript)
        if alignments:
            logger.info("Sentence-level wav2vec2 fallback succeeded")
            return alignments
    except Exception as exc:
        logger.warning("Sentence-level fallback failed (%s)", exc)

    # Path 4 – CMUdict ASR word comparison
    logger.warning("All phoneme backends failed; using ASR-word CMUdict fallback")
    return _asr_word_fallback(transcript, asr_transcript)


# ---------------------------------------------------------------------------
# Path 1 – MFA
# ---------------------------------------------------------------------------

def _mfa_align(audio_path: str | Path, transcript: str) -> List[WordAlignment]:
    import textgrid

    audio_path = Path(audio_path)
    work_dir   = Path(tempfile.mkdtemp(prefix="mfa_"))
    try:
        corpus_dir = work_dir / "corpus"
        corpus_dir.mkdir()
        output_dir = work_dir / "output"
        output_dir.mkdir()

        shutil.copy(audio_path, corpus_dir / "utterance.wav")
        (corpus_dir / "utterance.txt").write_text(transcript, encoding="utf-8")

        cmd = [
            "mfa", "align",
            str(corpus_dir), MFA_DICTIONARY, MFA_ACOUSTIC_MODEL,
            str(output_dir), "--clean",
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        tg = textgrid.TextGrid.fromFile(str(output_dir / "utterance.TextGrid"))
        return _parse_textgrid(tg)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _parse_textgrid(tg) -> List[WordAlignment]:
    word_tier    = next(t for t in tg.tiers if t.name.lower() == "words")
    phoneme_tier = next(t for t in tg.tiers if t.name.lower() in ("phones", "phonemes"))
    alignments   = []
    for w in word_tier:
        label = w.mark.strip()
        if not label or label in ("<eps>", "sp"):
            continue
        ph_data = []
        for p in phoneme_tier:
            if p.maxTime <= w.minTime:
                continue
            if p.minTime >= w.maxTime:
                break
            ph_label = p.mark.strip().rstrip("012")
            if ph_label and ph_label not in ("<eps>", "sp"):
                ph_data.append({
                    "phoneme": ph_label, "start": p.minTime,
                    "end": p.maxTime, "confidence": 1.0,
                })
        alignments.append(build_word_alignment(
            word=label, start=w.minTime, end=w.maxTime, phoneme_data=ph_data
        ))
    return alignments


# ---------------------------------------------------------------------------
# Path 2 – Per-word audio slicing using Whisper timestamps
# ---------------------------------------------------------------------------

def _load_full_audio(audio_path: str | Path) -> Tuple[np.ndarray, int]:
    """Load audio as float32 mono at 16 kHz using phoneme_recognition_service loader."""
    audio_np = phoneme_recognition_service._load_audio_as_float32(audio_path)
    return audio_np, SAMPLE_RATE


def _slice_audio(
    audio: np.ndarray,
    sr: int,
    start: float,
    end: float,
) -> np.ndarray:
    """Extract a word-level clip from the full waveform with padding."""
    duration = len(audio) / sr
    s = max(0.0, start - CLIP_PAD_SEC)
    e = min(duration, end + CLIP_PAD_SEC)
    s_idx = int(s * sr)
    e_idx = int(e * sr)
    clip = audio[s_idx:e_idx]
    # If clip is too short, pad with zeros
    min_samples = int(MIN_CLIP_SEC * sr)
    if len(clip) < min_samples:
        clip = np.pad(clip, (0, min_samples - len(clip)))
    return clip


def _save_clip_to_wav(clip: np.ndarray, sr: int) -> str:
    """Write a float32 numpy array to a temp WAV file, return path."""
    import scipy.io.wavfile as wf
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    # scipy.io.wavfile wants int16
    pcm = (clip * 32767).clip(-32768, 32767).astype(np.int16)
    wf.write(tmp.name, sr, pcm)
    return tmp.name


def _match_timestamps_to_words(
    correct_words: List[str],
    timestamps: List[WordTimestamp],
) -> List[Optional[WordTimestamp]]:
    """
    Match Whisper word timestamps to the correct (confirmed) word list.

    Whisper may produce slightly different tokenisation (e.g. "I'll" vs
    "I" + "'ll"), so we do a best-effort fuzzy match by cleaning both sides.
    Returns a list of the same length as correct_words, with None where no
    timestamp was found.
    """
    def _clean(w: str) -> str:
        return _PUNCT.sub("", w).strip().lower()

    # Build a queue of available timestamps
    ts_queue = list(timestamps)
    matched: List[Optional[WordTimestamp]] = []

    for cw in correct_words:
        cw_clean = _clean(cw)
        found: Optional[WordTimestamp] = None

        # Try exact match first, then prefix/contains match
        for i, ts in enumerate(ts_queue):
            ts_clean = _clean(ts.word)
            if ts_clean == cw_clean or ts_clean.startswith(cw_clean) or cw_clean.startswith(ts_clean):
                found = ts_queue.pop(i)
                break

        matched.append(found)

    return matched


def _per_word_slice_align(
    audio_path: str | Path,
    transcript: str,
    word_timestamps: List[WordTimestamp],
) -> List[WordAlignment]:
    """
    Main fallback: use Whisper timestamps to slice audio per word,
    then run wav2vec2 phoneme recognition on each clip.
    """
    correct_words = transcript.split()
    audio, sr     = _load_full_audio(audio_path)
    duration      = len(audio) / sr

    # Match timestamps to correct words
    matched = _match_timestamps_to_words(correct_words, word_timestamps)

    alignments: List[WordAlignment] = []
    tmp_files: List[str] = []

    try:
        for word, ts in zip(correct_words, matched):
            expected = get_expected_phonemes(word)

            if ts is None or ts.end - ts.start < 0.05:
                # No timestamp for this word – give full credit to avoid
                # unfair 0/100 for words Whisper simply didn't timestamp
                logger.debug("No timestamp for word='%s'; using expected as detected", word)
                ph_data = [
                    {"phoneme": ph, "start": 0.0, "end": 0.0, "confidence": 0.85}
                    for ph in expected
                ]
                alignments.append(build_word_alignment(
                    word=word, start=0.0, end=0.0, phoneme_data=ph_data
                ))
                continue

            # Slice and save the word's audio clip
            clip     = _slice_audio(audio, sr, ts.start, ts.end)
            clip_path = _save_clip_to_wav(clip, sr)
            tmp_files.append(clip_path)

            # Run phoneme recognition on the clip
            detected = phoneme_recognition_service.recognize_phonemes(clip_path)

            if detected is None:
                # Model unavailable – use expected phonemes (no error detection)
                detected_ph = expected
                conf        = 0.85
            elif len(detected) == 0:
                # Model ran but heard nothing (very short / silent clip)
                detected_ph = []
                conf        = 0.5
            else:
                detected_ph = detected
                conf        = 0.8

            ph_data = [
                {"phoneme": ph, "start": ts.start, "end": ts.end, "confidence": conf}
                for ph in detected_ph
            ]
            alignments.append(build_word_alignment(
                word=word,
                start=ts.start,
                end=ts.end,
                phoneme_data=ph_data,
            ))
            logger.info(
                "word='%s'  ts=[%.2f,%.2f]  expected=%s  detected=%s",
                word, ts.start, ts.end, expected, detected_ph,
            )

    finally:
        for f in tmp_files:
            try:
                os.unlink(f)
            except Exception:
                pass

    return alignments


# ---------------------------------------------------------------------------
# Path 3 – Full-sentence wav2vec2 + proportional distribution (short sentences)
# ---------------------------------------------------------------------------

def _sentence_level_fallback(
    audio_path: str | Path,
    transcript: str,
) -> List[WordAlignment]:
    """
    Run wav2vec2 on the full sentence audio and distribute phonemes
    proportionally across words.  Only reasonably reliable for short
    sentences (≤ 8 words).
    """
    words            = transcript.split()
    all_detected     = phoneme_recognition_service.recognize_phonemes(audio_path)

    if all_detected is None or not all_detected:
        return []

    expected_per_word = [get_expected_phonemes(w) for w in words]
    total_expected    = sum(len(e) for e in expected_per_word)
    if total_expected == 0:
        return []

    alignments      = []
    detected_cursor = 0

    for idx, (word, expected) in enumerate(zip(words, expected_per_word)):
        n_expected = len(expected)
        if n_expected == 0:
            alignments.append(build_word_alignment(
                word=word, start=0.0, end=0.0, phoneme_data=[]))
            continue

        remaining_det = len(all_detected) - detected_cursor
        remaining_exp = sum(len(e) for e in expected_per_word[idx:])
        proportion    = n_expected / max(remaining_exp, 1)
        n_assign      = max(1, round(proportion * remaining_det))
        n_assign      = min(n_assign, remaining_det)

        word_detected = all_detected[detected_cursor: detected_cursor + n_assign]
        detected_cursor += n_assign

        ph_data = [
            {"phoneme": ph, "start": 0.0, "end": 0.0, "confidence": 0.75}
            for ph in word_detected
        ]
        alignments.append(build_word_alignment(
            word=word, start=0.0, end=0.0, phoneme_data=ph_data
        ))

    return alignments


# ---------------------------------------------------------------------------
# Path 4 – CMUdict ASR-word fallback
# ---------------------------------------------------------------------------

def _asr_word_fallback(
    correct_transcript: str,
    asr_transcript: Optional[str],
) -> List[WordAlignment]:
    """
    Map each Whisper-heard word to its CMUdict phonemes as the 'detected'
    sequence.  Whisper may normalise mild mispronunciations, so substitution
    errors are not detected here — but at least scores won't be 0/100.
    """
    correct_words = correct_transcript.split()
    asr_words     = asr_transcript.split() if asr_transcript else []
    use_asr       = len(asr_words) == len(correct_words)

    alignments = []
    for idx, cw in enumerate(correct_words):
        heard = asr_words[idx] if use_asr else cw
        det   = get_expected_phonemes(heard)
        ph_data = [
            {"phoneme": ph, "start": 0.0, "end": 0.0, "confidence": 0.7}
            for ph in det
        ]
        alignments.append(build_word_alignment(
            word=cw, start=0.0, end=0.0, phoneme_data=ph_data
        ))
    return alignments
