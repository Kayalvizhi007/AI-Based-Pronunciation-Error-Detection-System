"""
Services Layer - Text to Speech
-------------------------------
Wraps pyttsx3 to speak feedback aloud.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def speak(text: str, rate: int = 160, volume: float = 0.9) -> None:
    """Speak text using local TTS engine."""
    logger.info("TTS speak requested (chars=%d, rate=%d, volume=%.2f)", len(text), rate, volume)
    try:
        import pyttsx3

        engine = pyttsx3.init()
        engine.setProperty("rate", rate)
        engine.setProperty("volume", volume)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
        logger.info("TTS playback completed")
    except Exception:
        logger.exception("TTS unavailable")


def speak_word(word: str) -> None:
    """Pronounce a single word with slower rate."""
    speak(word, rate=120)


def speak_phoneme(phoneme: str) -> None:
    """Pronounce a phoneme label."""
    speak(phoneme, rate=100)
