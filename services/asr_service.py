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

_model_size       = os.getenv("WHISPER_MODEL", "tiny.en").split("#", 1)[0].strip()
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

def _preprocess_for_asr(audio_path: str | Path) -> str:
    """Return a (possibly temporary) WAV file path suitable for ASR.

    This allows us to normalise and run the same lightweight noise reduction we
    applied during phoneme recognition.  Whisper-style models accept a file
    path, so we write the cleaned audio to a temp file and pass that along.  If
    the preprocessing helpers are unavailable the original path is returned.
    """
    try:
        from infrastructure.audio_processing import load_wav, SAMPLE_RATE
        import scipy.io.wavfile as wavfile

        audio = load_wav(audio_path)
        # convert float32 [-1,1] back to int16 for WAV encoding
        int16 = (audio * 32767).astype("int16")
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        wavfile.write(tmp.name, SAMPLE_RATE, int16)
        return tmp.name
    except Exception:
        # log at debug level but don't crash if something goes wrong
        logger.debug("ASR preprocessing skipped for %s", audio_path)
        return str(audio_path)


def transcribe_audio(audio_path: str | Path) -> str:
    """Transcribe audio and return plain text."""
    model, backend = _get_model()
    preprocessed = _preprocess_for_asr(audio_path)
    logger.info("Transcribing via %s: %s", backend, preprocessed)
    t0 = time.perf_counter()

    try:
        if backend == "faster_whisper":
            # Greedy decoding is faster and usually sufficient for short phrases
            segments, _ = model.transcribe(
                str(preprocessed),
                language="en",
                beam_size=1,
                best_of=1,
            )
            text = " ".join(s.text.strip() for s in segments).strip()
        else:
            # openai-whisper is slower; keep fp16 off for CPU
            result = model.transcribe(str(preprocessed), language="en", fp16=False)
            text = result["text"].strip()
    finally:
        # remove temporary file if we created one
        if preprocessed != str(audio_path):
            try:
                os.unlink(preprocessed)
            except Exception:
                logger.debug("Failed to remove temp ASR file %s", preprocessed)

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

    preprocessed = _preprocess_for_asr(audio_path)
    try:
        if backend == "faster_whisper":
            text, words = _timestamps_faster_whisper(model, str(preprocessed))
        else:
            text, words = _timestamps_openai_whisper(model, str(preprocessed))

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
    finally:
        if preprocessed != str(audio_path):
            try:
                os.unlink(preprocessed)
            except Exception:
                logger.debug("Failed to remove temp ASR file %s", preprocessed)


def _timestamps_faster_whisper(
    model, audio_path: str
) -> Tuple[str, List[WordTimestamp]]:
    segments, _ = model.transcribe(
        audio_path,
        language="en",
        word_timestamps=True,   # ← key flag
        beam_size=1,
        best_of=1,
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
