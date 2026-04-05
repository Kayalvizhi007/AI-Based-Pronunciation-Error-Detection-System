"""
Services Layer – TTS Audio Generation
---------------------------------------
Generates audio bytes for a word or sentence so Streamlit can play them
via st.audio() without needing a system speaker or pyttsx3.

Uses gTTS (Google Text-to-Speech) which returns an MP3 BytesIO object.
Falls back to pyttsx3 + a temp file if gTTS is unavailable (offline).

Install:
    pip install gTTS
"""

from __future__ import annotations

import io
import logging
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def word_audio_bytes(word: str) -> Optional[bytes]:
    """
    Return MP3 audio bytes for a single word spoken clearly.
    Returns None if both backends fail.
    """
    return _generate_bytes(word, slow=True)


def sentence_audio_bytes(sentence: str) -> Optional[bytes]:
    """Return MP3 audio bytes for a full sentence."""
    return _generate_bytes(sentence, slow=False)


def _generate_bytes(text: str, slow: bool = False) -> Optional[bytes]:
    # --- Try gTTS first (internet required) ---
    try:
        from gtts import gTTS
        buf = io.BytesIO()
        gTTS(text=text, lang="en", slow=slow).write_to_fp(buf)
        buf.seek(0)
        logger.debug("gTTS generated audio for %r", text)
        return buf.read()
    except Exception as gtts_err:
        logger.warning("gTTS failed (%s); trying pyttsx3 fallback", gtts_err)

    # --- Fallback: pyttsx3 → temp file → read bytes ---
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 120 if slow else 160)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        engine.save_to_file(text, tmp_path)
        engine.runAndWait()
        engine.stop()
        data = Path(tmp_path).read_bytes()
        Path(tmp_path).unlink(missing_ok=True)
        logger.debug("pyttsx3 generated audio for %r", text)
        return data
    except Exception as pyttsx_err:
        logger.error("pyttsx3 fallback also failed (%s); no audio available", pyttsx_err)
        return None
