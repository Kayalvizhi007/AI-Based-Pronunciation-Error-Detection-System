"""
Infrastructure Layer - Audio Processing
---------------------------------------
Utilities for recording, loading, and normalizing audio.
"""

from __future__ import annotations

import logging
import tempfile
import wave
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
CHANNELS = 1
DTYPE = "int16"
DEFAULT_SECS = 5


def preprocess_audio(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Run the full preprocessing pipeline on raw audio.

    Steps:
    1. Convert to float32 and normalise.
    2. Apply noise reduction.
    3. Apply a speech-friendly bandpass filter (80-8000 Hz).
    4. Trim leading/trailing silence.
    5. Apply adaptive gain if the signal is too quiet.
    """
    audio = audio.astype(np.float32)
    audio = normalise(audio)

    audio = denoise(audio, sr=sr)
    audio = bandpass_filter(audio, sr)
    audio = trim_silence(audio)
    audio = apply_auto_gain(audio)

    return audio.astype(np.float32)


def record_audio(duration: int = DEFAULT_SECS) -> np.ndarray:
    """Record audio from default microphone."""
    import sounddevice as sd

    logger.info("Recording audio for %d seconds", duration)
    frames = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
    )
    sd.wait()
    flattened = frames.flatten().astype(np.float32)

    processed = preprocess_audio(flattened, sr=SAMPLE_RATE)
    logger.info("Recording complete (samples=%d)", len(processed))
    return processed


def record_to_file(path: str | Path, duration: int = DEFAULT_SECS) -> Path:
    """Record and write WAV file."""
    audio = record_audio(duration)
    path = Path(path)
    _save_wav(path, audio)
    logger.info("Audio recorded to file: %s", path)
    return path


def record_to_tempfile(duration: int = DEFAULT_SECS) -> Path:
    """Record audio to a temporary WAV file."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    return record_to_file(tmp.name, duration)


def _save_wav(path: Path, audio: np.ndarray) -> None:
    # Whisper and most audio tools expect 16-bit PCM WAV files.
    # Our internal pipeline works in float32 [-1, 1], so convert here.
    audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(audio_int16.tobytes())


def bytes_to_wav_file(audio_bytes: bytes, suffix: str = ".wav") -> Path:
    """Write audio bytes to a temporary file."""
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(audio_bytes)
    tmp.close()
    logger.debug("Wrote %d bytes to temp wav: %s", len(audio_bytes), tmp.name)
    return Path(tmp.name)


def load_wav(path: str | Path) -> np.ndarray:
    """Load WAV, normalize, clean, and return float32 audio suitable for ASR."""
    import scipy.io.wavfile as wavfile

    rate, data = wavfile.read(str(path))
    logger.debug("Loaded wav file: %s (rate=%s, samples=%d)", path, rate, len(data))

    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    if data.ndim > 1:
        data = data.mean(axis=1)

    # If the file isn't at our target rate, resample
    if rate != SAMPLE_RATE:
        try:
            from scipy.signal import resample
            num = int(len(data) * SAMPLE_RATE / rate)
            data = resample(data, num).astype(np.float32)
        except Exception:
            logger.warning("Could not resample audio (%s Hz); leaving as-is", rate)

    return preprocess_audio(data, sr=SAMPLE_RATE)


def normalise(audio: np.ndarray) -> np.ndarray:
    """Peak-normalize audio array."""
    peak = np.max(np.abs(audio))
    if peak == 0:
        return audio
    return audio / peak


def trim_silence(
    audio: np.ndarray,
    threshold: float = 0.005,
    min_silence_duration: float = 0.08,
) -> np.ndarray:
    """Trim leading/trailing silence from an audio clip.

    This helps focus on the spoken portion of the recording and reduces the
    chance of capturing background noise before/after speech.
    """
    if audio.size == 0:
        return audio

    abs_audio = np.abs(audio)
    mask = abs_audio > threshold
    if not np.any(mask):
        return audio

    start = int(np.argmax(mask))
    end = len(mask) - int(np.argmax(mask[::-1]))

    min_samples = int(min_silence_duration * SAMPLE_RATE)
    if end - start < min_samples:
        # too short after trimming – keep original to avoid chopping speech
        return audio

    return audio[start:end].astype(np.float32)


def bandpass_filter(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Apply a bandpass filter that keeps the human speech band (80–8000 Hz)."""
    try:
        from scipy.signal import butter, filtfilt
    except ImportError:
        return audio

    low, high = 80.0, 8000.0
    nyq = sr / 2.0
    if high >= nyq:
        high = nyq - 1.0
    b, a = butter(4, [low / nyq, high / nyq], btype="band")
    try:
        return filtfilt(b, a, audio)
    except Exception:
        return audio


def apply_auto_gain(audio: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
    """Apply automatic gain control to bring RMS level closer to target."""
    rms = np.sqrt(np.mean(audio**2))
    if rms < 1e-6:
        return audio
    gain = target_rms / rms
    # Avoid extreme gain that amplifies noise
    gain = np.clip(gain, 0.5, 5.0)
    return np.clip(audio * gain, -1.0, 1.0)


def denoise(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Apply a simple noise-reduction filter to the waveform.

    First attempt to use :mod:`noisereduce` if it is installed, falling back to
    a Wiener filter from :mod:`scipy.signal`.  The goal is to suppress
    background noise so that downstream phoneme extraction is more reliable in
    noisy environments.

    The function is intentionally lightweight; advanced users may replace it
    with a more sophisticated pipeline later.
    """
    # leave the original array untouched if noise reduction isn't possible
    try:
        import noisereduce as nr
        cleaned = nr.reduce_noise(y=audio.astype(np.float32), sr=sr)
    except ImportError:
        logger.debug("noisereduce not available; falling back to scipy.wiener")
        try:
            from scipy.signal import wiener
            # if the signal is constant there is nothing to denoise; avoid warnings
            if np.std(audio) == 0:
                cleaned = audio
            else:
                cleaned = wiener(audio).astype(audio.dtype)
        except Exception as e:  # scipy missing or filter failed
            logger.warning("noise reduction failed (%s); returning original audio", e)
            cleaned = audio

    # Trim silence after denoising to remove long leading/trailing noise
    try:
        cleaned = trim_silence(cleaned)
    except Exception:
        pass

    return cleaned
