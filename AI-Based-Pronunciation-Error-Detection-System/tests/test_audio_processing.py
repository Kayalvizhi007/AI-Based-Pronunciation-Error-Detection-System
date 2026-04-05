"""
Unit tests for audio processing utilities.
"""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from infrastructure import audio_processing


def test_normalise_preserves_shape_and_scales():
    arr = np.array([0.0, 1.0, -2.0, 0.5], dtype=np.float32)
    result = audio_processing.normalise(arr)
    assert result.shape == arr.shape
    assert np.max(np.abs(result)) == pytest.approx(1.0)


def test_trim_silence_removes_leading_trailing_noise():
    # audio with quiet leading/trailing sections and a loud center tone
    silence = np.zeros(100, dtype=np.float32)
    tone = np.ones(200, dtype=np.float32)
    audio = np.concatenate([silence, tone, silence])

    trimmed = audio_processing.trim_silence(audio, threshold=0.1, min_silence_duration=0.01)
    assert trimmed.shape[0] == 200


def test_denoise_reduces_noise():
    # create a clean sine wave and add white noise
    rng = np.random.default_rng(12345)
    t = np.linspace(0, 1, audio_processing.SAMPLE_RATE, endpoint=False)
    clean = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    noisy = clean + 0.5 * rng.standard_normal(clean.shape, dtype=np.float32)

    denoised = audio_processing.denoise(noisy)
    assert denoised.shape == noisy.shape
    # denoised signal should be closer to the clean signal than the noisy one
    orig_err = np.linalg.norm(noisy - clean)
    new_err = np.linalg.norm(denoised - clean)
    assert new_err <= orig_err


@pytest.mark.parametrize("duration", [1, 2])
def test_record_audio_returns_array(monkeypatch, duration):
    # monkeypatch sounddevice to generate dummy signal instead of recording
    class DummyRec:
        @staticmethod
        def rec(frames, samplerate, channels, dtype):
            return np.ones((frames, channels), dtype=dtype)

        @staticmethod
        def wait():
            pass

    monkeypatch.setitem(__import__("sys").modules, "sounddevice", DummyRec)
    audio = audio_processing.record_audio(duration=duration)
    assert audio.dtype == np.float32
    assert abs(audio.size - duration * audio_processing.SAMPLE_RATE) <= 1


def test_preprocess_for_asr_creates_file(tmp_path):
    """Verify that ASR preprocessing loads and writes a cleaned WAV file."""
    import scipy.io.wavfile as wavfile

    # generate a simple sine wave file
    rate = audio_processing.SAMPLE_RATE
    t = np.linspace(0, 0.1, rate // 10, endpoint=False)
    data = (0.5 * np.sin(2 * np.pi * 440 * t)).astype("float32")
    wav_path = tmp_path / "input.wav"
    wavfile.write(wav_path, rate, (data * 32767).astype("int16"))

    from services import asr_service
    proc_path = asr_service._preprocess_for_asr(wav_path)
    assert proc_path != str(wav_path)
    assert Path(proc_path).exists()
    # loaded audio should have similar length to the original
    r2, d2 = wavfile.read(proc_path)
    assert r2 == rate
    assert abs(d2.shape[0] - data.shape[0]) <= 2


def test_save_wav_writes_int16(tmp_path):
    """Ensure saved WAV files are written as 16-bit PCM, not float32."""
    import scipy.io.wavfile as wavfile

    audio = np.array([0.0, 0.5, -0.5], dtype=np.float32)
    path = tmp_path / "out.wav"
    audio_processing._save_wav(path, audio)

    rate, data = wavfile.read(path)
    assert rate == audio_processing.SAMPLE_RATE
    assert data.dtype == "int16"
    assert data.max() <= 32767


def test_transcribe_audio_cleans_temp(tmp_path, monkeypatch):
    # create fake wav file
    import scipy.io.wavfile as wavfile
    rate = audio_processing.SAMPLE_RATE
    data = np.zeros(rate, dtype="int16")
    orig = tmp_path / "orig.wav"
    wavfile.write(orig, rate, data)

    # monkeypatch preprocessing to return a new temp path
    fake_tmp = tmp_path / "temp.wav"
    fake_tmp.write_bytes(b"")
    monkeypatch.setattr(
        "services.asr_service._preprocess_for_asr", lambda p: str(fake_tmp)
    )
    # monkeypatch model to avoid real transcription
    class DummyModel:
        def transcribe(self, path, *args, **kwargs):
            # Accept extra keyword args used by the faster-whisper call
            if kwargs.get("word_timestamps"):
                return ([], [])
            return ([], None)
    monkeypatch.setattr("services.asr_service._get_model", lambda: (DummyModel(), "faster_whisper"))

    # call and verify temp file removed
    from services import asr_service
    asr_service.transcribe_audio(orig)
    assert not fake_tmp.exists()
