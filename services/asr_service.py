"""
Services Layer - ASR (Automatic Speech Recognition)
----------------------------------------------------
Provides speech-to-text with backend fallback.
Also exposes transcribe_with_word_timestamps() which returns per-word
timing so the phoneme recognition service can slice audio per word.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

_model = None
_backend_in_use = None

_model_size       = os.getenv("WHISPER_MODEL", "base").split("#", 1)[0].strip()
_preferred_backend = os.getenv("ASR_BACKEND", "faster_whisper").strip().lower()
_device           = os.getenv("WHISPER_DEVICE", "cpu").strip().lower()
_compute_type     = os.getenv("WHISPER_COMPUTE_TYPE", "int8").strip().lower()


@dataclass
class WordTimestamp:
    """A single word with its start/end time in seconds."""
    word:  str
    start: float
    end:   float


def _load_faster_whisper():
    from faster_whisper import WhisperModel
    logger.info(
        "Loading faster-whisper model='%s' device='%s' compute_type='%s'",
        _model_size, _device, _compute_type,
    )
    return WhisperModel(_model_size, device=_device, compute_type=_compute_type)


def _load_openai_whisper():
    import whisper
    logger.info("Loading openai-whisper model='%s'", _model_size)
    return whisper.load_model(_model_size)


def _get_model():
    global _model, _backend_in_use
    if _model is not None:
        return _model, _backend_in_use

    loaders = {
        "faster_whisper": _load_faster_whisper,
        "openai_whisper": _load_openai_whisper,
    }

    backend_order = (
        ["faster_whisper", "openai_whisper"]
        if _preferred_backend != "openai_whisper"
        else ["openai_whisper", "faster_whisper"]
    )

    last_error = None
    for name in backend_order:
        try:
            started = time.perf_counter()
            model   = loaders[name]()
            _model, _backend_in_use = model, name
            logger.info("ASR ready: %s (%.2fs)", name, time.perf_counter() - started)
            return _model, _backend_in_use
        except Exception as exc:
            last_error = exc
            logger.exception("ASR backend '%s' failed: %s", name, exc)

    raise RuntimeError("No ASR backend available.") from last_error


# ---------------------------------------------------------------------------
# Plain transcription (unchanged public API)
# ---------------------------------------------------------------------------

def transcribe_audio(audio_path: str | Path) -> str:
    """Transcribe audio and return plain text."""
    model, backend = _get_model()
    logger.info("Transcribing via %s: %s", backend, audio_path)
    t0 = time.perf_counter()

    if backend == "faster_whisper":
        segments, _ = model.transcribe(str(audio_path), language="en")
        text = " ".join(s.text.strip() for s in segments).strip()
    else:
        result = model.transcribe(str(audio_path), language="en", fp16=False)
        text = result["text"].strip()

    logger.info("Transcription done in %.2fs: %r", time.perf_counter() - t0, text)
    return text


def transcribe_bytes(audio_bytes: bytes, suffix: str = ".wav") -> str:
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        return transcribe_audio(tmp_path)
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# NEW: transcription + word-level timestamps
# ---------------------------------------------------------------------------

def transcribe_with_word_timestamps(
    audio_path: str | Path,
) -> Tuple[str, List[WordTimestamp]]:
    """
    Transcribe audio and return (full_text, [WordTimestamp(word, start, end), ...]).

    Word timestamps allow the phoneme service to slice the audio and run
    phoneme recognition on each word independently, which is far more accurate
    than running on the full sentence and distributing phonemes proportionally.

    Falls back gracefully: if timestamps cannot be extracted, returns the
    plain transcript with empty word list.
    """
    model, backend = _get_model()
    logger.info("Transcribing with word timestamps via %s: %s", backend, audio_path)
    t0 = time.perf_counter()

    try:
        if backend == "faster_whisper":
            text, words = _timestamps_faster_whisper(model, str(audio_path))
        else:
            text, words = _timestamps_openai_whisper(model, str(audio_path))

        logger.info(
            "Timestamps done in %.2fs: %d words, text=%r",
            time.perf_counter() - t0, len(words), text,
        )
        return text, words

    except Exception as exc:
        logger.warning(
            "Word timestamp extraction failed (%s); falling back to plain transcription", exc
        )
        text = transcribe_audio(audio_path)
        return text, []


def _timestamps_faster_whisper(
    model, audio_path: str
) -> Tuple[str, List[WordTimestamp]]:
    segments, _ = model.transcribe(
        audio_path,
        language="en",
        word_timestamps=True,   # ← key flag
    )
    words: List[WordTimestamp] = []
    parts: List[str] = []
    for seg in segments:
        parts.append(seg.text.strip())
        if seg.words:
            for w in seg.words:
                words.append(WordTimestamp(
                    word=w.word.strip(),
                    start=w.start,
                    end=w.end,
                ))
    return " ".join(parts).strip(), words


def _timestamps_openai_whisper(
    model, audio_path: str
) -> Tuple[str, List[WordTimestamp]]:
    result = model.transcribe(
        audio_path,
        language="en",
        fp16=False,
        word_timestamps=True,
    )
    words: List[WordTimestamp] = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            words.append(WordTimestamp(
                word=w["word"].strip(),
                start=w["start"],
                end=w["end"],
            ))
    return result["text"].strip(), words
