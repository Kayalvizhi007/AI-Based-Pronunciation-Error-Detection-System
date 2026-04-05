"""
Services Layer - LLM Interface
------------------------------
Supports online Gemini and local Ollama backends with safe fallbacks.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
import warnings
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

_gemini_client = None
_gemini_legacy_mode = False
_gemini_disabled_until = 0.0
_env_checked = False

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_OLLAMA_MODEL = "qwen2.5:1.5b"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"


def _llm_backend() -> str:
    # auto | ollama | gemini
    return os.getenv("LLM_BACKEND", "auto").strip().lower()


def _gemini_model_name() -> str:
    return os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)


def _ollama_model_name() -> str:
    return os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)


def _ollama_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL).rstrip("/")


def _ollama_timeout_sec() -> int:
    try:
        return int(os.getenv("OLLAMA_TIMEOUT_SEC", "120"))
    except ValueError:
        return 120


def _ensure_env_loaded() -> None:
    global _env_checked
    if _env_checked:
        return
    _env_checked = True
    try:
        from dotenv import load_dotenv

        env_path = Path(__file__).resolve().parents[1] / ".env"
        loaded = load_dotenv(env_path)
        logger.info("Dotenv load attempted from %s (loaded=%s)", env_path, loaded)
    except Exception:
        logger.exception("Failed while attempting to load .env")


def _get_gemini_client():
    global _gemini_client, _gemini_legacy_mode
    if _gemini_client is not None:
        return _gemini_client

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        _ensure_env_loaded()
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY (or GOOGLE_API_KEY) environment variable is not set."
        )

    try:
        from google import genai  # pip install google-genai

        _gemini_client = genai.Client(api_key=api_key)
        _gemini_legacy_mode = False
        logger.info(
            "Initialized Gemini client with google-genai (model=%s)",
            _gemini_model_name(),
        )
        return _gemini_client
    except ImportError:
        logger.warning("google-genai not installed; falling back to google-generativeai")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            import google.generativeai as legacy_genai
    except ImportError as exc:
        raise ImportError(
            "Neither `google-genai` nor `google-generativeai` is installed."
        ) from exc

    legacy_genai.configure(api_key=api_key)
    _gemini_client = legacy_genai.GenerativeModel(_gemini_model_name())
    _gemini_legacy_mode = True
    logger.info(
        "Initialized Gemini client with legacy google-generativeai (model=%s)",
        _gemini_model_name(),
    )
    return _gemini_client


def _generate_with_gemini(prompt: str, task_name: str) -> str:
    global _gemini_disabled_until
    if time.time() < _gemini_disabled_until:
        wait = int(_gemini_disabled_until - time.time())
        raise RuntimeError(f"Gemini temporarily disabled due to quota. Retry after ~{wait}s.")

    client = _get_gemini_client()
    logger.info("Calling Gemini for task=%s", task_name)

    if _gemini_legacy_mode:
        response = client.generate_content(prompt)
    else:
        response = client.models.generate_content(
            model=_gemini_model_name(),
            contents=prompt,
        )

    text = getattr(response, "text", None)
    if not text:
        raise RuntimeError(f"Gemini returned empty response for task={task_name}")

    logger.info("Gemini response received for task=%s (chars=%d)", task_name, len(text))
    return text.strip()


def _gemini_backoff_seconds(exc: Exception) -> int:
    message = str(exc)
    match = re.search(r"retry in ([0-9]+(?:\\.[0-9]+)?)s", message, flags=re.IGNORECASE)
    if match:
        return max(10, int(float(match.group(1))))
    return 60


def _generate_with_ollama(prompt: str, task_name: str) -> str:
    url = f"{_ollama_base_url()}/api/generate"
    payload = {
        "model": _ollama_model_name(),
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2},
    }
    request = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    logger.info(
        "Calling Ollama for task=%s (model=%s, url=%s)",
        task_name,
        _ollama_model_name(),
        url,
    )
    with urllib.request.urlopen(request, timeout=_ollama_timeout_sec()) as response:
        body = json.loads(response.read().decode("utf-8"))

    text = (body.get("response") or "").strip()
    if not text:
        raise RuntimeError(f"Ollama returned empty response for task={task_name}")

    logger.info("Ollama response received for task=%s (chars=%d)", task_name, len(text))
    return text


def _generate(prompt: str, task_name: str) -> str:
    global _gemini_disabled_until
    backend = _llm_backend()
    if backend == "ollama":
        return _generate_with_ollama(prompt, task_name)
    if backend == "gemini":
        return _generate_with_gemini(prompt, task_name)

    # auto mode: Gemini first (better quality), Ollama as fallback when Gemini
    # quota is exhausted or API key is missing.
    last_error = None
    for engine in ("gemini", "ollama"):
        try:
            if engine == "ollama":
                return _generate_with_ollama(prompt, task_name)
            return _generate_with_gemini(prompt, task_name)
        except Exception as exc:
            last_error = exc
            if engine == "gemini" and ("429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc)):
                backoff = _gemini_backoff_seconds(exc)
                _gemini_disabled_until = time.time() + backoff
                logger.warning("Gemini quota reached. Disabling Gemini for %ss.", backoff)
            logger.warning("%s backend failed for task=%s: %s", engine, task_name, exc)
    raise RuntimeError(f"All LLM backends failed for task={task_name}") from last_error


def _heuristic_correction(text: str) -> str:
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return cleaned
    return cleaned[0].upper() + cleaned[1:]


def _rule_based_explanation(errors: List[dict], word: str) -> str:
    if not errors:
        return f"Great job! Your pronunciation of '{word}' sounds correct."
    first = errors[0]
    expected = first.get("expected_phoneme", "?")
    detected = first.get("detected_phoneme", "?")
    return (
        f"For '{word}', target /{expected}/ instead of /{detected}/. "
        "Slow down, exaggerate the mouth shape, and repeat the word 5 times."
    )


def _rule_based_summary(session_data: dict) -> str:
    word_results = session_data.get("word_results", []) or []
    total = len(word_results)
    passed = sum(1 for row in word_results if row.get("passed"))
    common = session_data.get("common_errors", []) or []
    common_text = ", ".join(common[:3]) if common else "none identified"
    return (
        f"Session complete. You passed {passed} out of {total} words. "
        f"Most frequent trouble sounds: {common_text}. "
        "Keep practicing slowly and focus on consistency over speed."
    )


def correct_transcript(raw_transcript: str) -> str:
    """Correct spelling/recognition mistakes in ASR output."""
    prompt = (
        "You are a spelling corrector for English speech transcripts. "
        "The following text was produced by a speech recognition system and may contain "
        "phonetic misspellings or recognition errors. "
        "Correct only clear spelling or recognition mistakes. "
        "Do NOT change the meaning or rephrase the sentence. "
        "Return ONLY the corrected sentence with no explanation.\n\n"
        f"Raw transcript: {raw_transcript}"
    )
    try:
        return _generate(prompt, task_name="correct_transcript")
    except Exception:
        logger.warning("LLM correction failed; using heuristic correction")
        return _heuristic_correction(raw_transcript)


def generate_explanation(errors: List[dict], word: str) -> str:
    """Generate a short tutor explanation for word-level pronunciation errors."""
    if not errors:
        return f"Great job! Your pronunciation of '{word}' sounds correct."

    payload = json.dumps(errors, indent=2)
    prompt = (
        "You are a friendly English pronunciation tutor. "
        "A learner is trying to pronounce the word below and made the following errors.\n\n"
        f"Word: {word}\n"
        f"Errors (JSON):\n{payload}\n\n"
        "Give a short, encouraging explanation (2-4 sentences) that:\n"
        "1. Names the specific sound(s) the learner got wrong.\n"
        "2. Describes mouth position, tongue placement, or airflow needed.\n"
        "3. Encourages the learner to try again.\n"
        "Speak directly to the learner ('you / your'). Keep it simple."
    )
    try:
        return _generate(prompt, task_name="generate_explanation")
    except Exception:
        logger.warning("LLM explanation failed; using rule-based explanation")
        return _rule_based_explanation(errors, word)


def generate_session_summary(session_data: dict) -> str:
    """Generate short coaching summary for completed session."""
    payload = json.dumps(session_data, indent=2)
    prompt = (
        "You are an English pronunciation coach. "
        "Below is a summary of a learner's pronunciation session (JSON).\n\n"
        f"{payload}\n\n"
        "Write a brief, encouraging session summary (3-5 sentences) that:\n"
        "1. Highlights words the learner did well on.\n"
        "2. Identifies the most common phoneme errors.\n"
        "3. Gives one or two focused practice tips.\n"
        "Keep the tone warm and motivating."
    )
    try:
        return _generate(prompt, task_name="generate_session_summary")
    except Exception:
        logger.warning("LLM summary failed; using rule-based summary")
        return _rule_based_summary(session_data)


# ---------------------------------------------------------------------------
# Health check – call this at startup or from the UI to verify backends
# ---------------------------------------------------------------------------

def check_backends() -> dict:
    """
    Test which LLM backends are reachable right now.

    Returns a dict like:
        {
            "gemini": True | False | "quota",
            "ollama": True | False,
            "rule_based": True,          # always available
        }

    Useful for displaying a status indicator in the UI.
    """
    result = {"gemini": False, "ollama": False, "rule_based": True}

    # --- Gemini ---
    try:
        if time.time() < _gemini_disabled_until:
            result["gemini"] = "quota"
        else:
            _get_gemini_client()   # just check init – no actual API call
            result["gemini"] = True
    except Exception as e:
        logger.debug("Gemini check failed: %s", e)
        result["gemini"] = False

    # --- Ollama ---
    try:
        url = f"{_ollama_base_url()}/api/tags"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=3):
            pass
        result["ollama"] = True
    except Exception as e:
        logger.debug("Ollama check failed: %s", e)
        result["ollama"] = False

    active = (
        "Gemini" if result["gemini"] is True
        else "Ollama" if result["ollama"]
        else "Rule-based fallback"
    )
    logger.info("LLM backend check: %s | active=%s", result, active)
    return result
