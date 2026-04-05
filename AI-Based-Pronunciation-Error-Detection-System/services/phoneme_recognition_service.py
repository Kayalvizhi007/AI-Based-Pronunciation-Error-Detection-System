"""
Services Layer – Phoneme Recognition
--------------------------------------
Uses vitouphy/wav2vec2-xls-r-300m-timit-phoneme to extract phonemes
from audio clips.

ROOT CAUSE FIX (this version)
------------------------------
The model's vocabulary uses IPA characters, not TIMIT text labels.
batch_decode concatenates them into strings like "haɪðɪs", "ɪs", "ðənɑ".
The old TIMIT→ARPAbet table (keyed on "hh", "ih" etc.) never matched them.

The fix has two parts:
  1. Use convert_ids_to_tokens() to get individual per-frame tokens BEFORE
     they are concatenated, so we can CTC-collapse them cleanly.
  2. Map each token to ARPAbet using a two-stage lookup:
       a. Direct ARPAbet match (handles models with uppercase ARPAbet vocab)
       b. IPA greedy longest-match parser (handles the IPA vocab this model uses)

IPA parser design
-----------------
Sort all known IPA→ARPAbet mappings longest-first and greedily consume the
token string from left to right.  This correctly splits multi-char IPA tokens
like "aɪ"→AY, "oʊ"→OW, "tʃ"→CH before single-char ones like "a"→AA.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_processor = None
_model      = None
_model_failed = False

MODEL_ID    = "vitouphy/wav2vec2-xls-r-300m-timit-phoneme"
SAMPLE_RATE = 16_000

# ---------------------------------------------------------------------------
# ARPAbet phoneme set (valid output labels)
# ---------------------------------------------------------------------------

_VALID_ARPABET = {
    "AA","AE","AH","AO","AW","AY",
    "B","CH","D","DH","EH","ER","EY",
    "F","G","HH","IH","IY","JH","K",
    "L","M","N","NG","OW","OY","P",
    "R","S","SH","T","TH","UH","UW",
    "V","W","Y","Z","ZH",
}

# ---------------------------------------------------------------------------
# IPA → ARPAbet greedy mapping
# Entries are (ipa_sequence, arpabet_label), sorted longest-first so greedy
# matching always prefers a longer match (e.g. "aɪ"→AY over "a"→AA).
# ---------------------------------------------------------------------------

_IPA_MAP_RAW = [
    # ── Diphthongs (2+ char IPA) — MUST come before component vowels ─────
    ("aɪ",  "AY"),
    ("aʊ",  "AW"),
    ("eɪ",  "EY"),
    ("oʊ",  "OW"),
    ("ɔɪ",  "OY"),
    # ── Affricates ────────────────────────────────────────────────────────
    ("tʃ",  "CH"),
    ("dʒ",  "JH"),
    ("ʧ",   "CH"),
    ("ʤ",   "JH"),
    # ── Rhotic vowel variants ─────────────────────────────────────────────
    ("ɝ",   "ER"),
    ("ɜ",   "ER"),
    ("ɚ",   "ER"),
    # ── Monophthong vowels ────────────────────────────────────────────────
    ("ɪ",   "IH"),
    ("i",   "IY"),
    ("ɛ",   "EH"),
    ("æ",   "AE"),
    ("ɑ",   "AA"),
    ("ɒ",   "AA"),   # British variant
    ("ɔ",   "AO"),
    ("ʊ",   "UH"),
    ("u",   "UW"),
    ("ə",   "AH"),
    ("ʌ",   "AH"),
    ("a",   "AA"),   # open front (fallback)
    ("e",   "EH"),   # close-mid front (fallback)
    ("o",   "OW"),   # close-mid back (fallback)
    # ── Consonants ────────────────────────────────────────────────────────
    ("ŋ",   "NG"),
    ("θ",   "TH"),
    ("ð",   "DH"),
    ("ʃ",   "SH"),
    ("ʒ",   "ZH"),
    ("ɹ",   "R"),
    ("ɾ",   "R"),    # alveolar flap (e.g. "butter")
    ("ʔ",   "T"),    # glottal stop
    ("h",   "HH"),
    ("b",   "B"),
    ("d",   "D"),
    ("f",   "F"),
    ("g",   "G"),
    ("k",   "K"),
    ("l",   "L"),
    ("m",   "M"),
    ("n",   "N"),
    ("p",   "P"),
    ("r",   "R"),
    ("s",   "S"),
    ("t",   "T"),
    ("v",   "V"),
    ("w",   "W"),
    ("j",   "Y"),
    ("z",   "Z"),
]

# Sort longest-first so greedy parse always prefers longer matches
_IPA_MAP_ORDERED = sorted(_IPA_MAP_RAW, key=lambda x: -len(x[0]))

# Pre-build a quick set of all IPA start characters for fast rejection
_IPA_START_CHARS = {seq[0] for seq, _ in _IPA_MAP_ORDERED}

# Skip tokens output by the model that carry no phoneme information
_SKIP_TOKENS = {"<pad>", "<s>", "</s>", "<unk>", "|", " ", "", "▁"}


# ---------------------------------------------------------------------------
# Model loading (lazy, cached)
# ---------------------------------------------------------------------------

def _get_model():
    global _processor, _model, _model_failed
    if _model_failed:
        raise RuntimeError("wav2vec2 model previously failed; skipping.")
    if _processor is not None:
        return _processor, _model
    try:
        from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
        logger.info("Loading phoneme model: %s", MODEL_ID)
        _processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
        _model     = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
        _model.eval()
        # Log a sample of the vocabulary so we can verify token format
        vocab = _processor.tokenizer.get_vocab()
        sample = list(vocab.keys())[:60]
        logger.info("Model vocab sample (first 60): %s", sample)
        logger.info("Phoneme model loaded OK")
        return _processor, _model
    except Exception as exc:
        _model_failed = True
        raise RuntimeError(f"Failed to load phoneme model: {exc}") from exc


# ---------------------------------------------------------------------------
# Audio loading (multi-backend, robust)
# ---------------------------------------------------------------------------

def _load_audio_as_float32(audio_path: str | Path) -> np.ndarray:
    """Load any audio file as float32 mono at 16 kHz."""
    path   = str(audio_path)
    errors = []

    # 1 – torchaudio
    try:
        import torchaudio
        waveform, sr = torchaudio.load(path)
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
        return waveform.mean(dim=0).numpy().astype(np.float32)
    except Exception as e:
        errors.append(f"torchaudio: {e}")

    # 2 – pydub + ffmpeg
    try:
        from pydub import AudioSegment
        seg = AudioSegment.from_file(path)
        seg = seg.set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(2)
        return np.frombuffer(seg.raw_data, dtype=np.int16).astype(np.float32) / 32768.0
    except Exception as e:
        errors.append(f"pydub: {e}")

    # 3 – soundfile
    try:
        import soundfile as sf
        import math
        from scipy.signal import resample_poly
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        mono = data.mean(axis=1)
        if sr != SAMPLE_RATE:
            gcd  = math.gcd(SAMPLE_RATE, sr)
            mono = resample_poly(mono, SAMPLE_RATE // gcd, sr // gcd).astype(np.float32)
        return mono
    except Exception as e:
        errors.append(f"soundfile: {e}")

    # 4 – scipy
    try:
        import scipy.io.wavfile as wf
        from scipy.signal import resample
        sr, data = wf.read(path)
        if data.ndim > 1:
            data = data.mean(axis=1)
        audio = data.astype(np.float32)
        if audio.max() > 1.0:
            audio /= 32768.0
        if sr != SAMPLE_RATE:
            audio = resample(audio, int(len(audio) * SAMPLE_RATE / sr)).astype(np.float32)
        return audio
    except Exception as e:
        errors.append(f"scipy: {e}")

    raise RuntimeError(f"Cannot load audio '{path}'. Tried: {'; '.join(errors)}")


# ---------------------------------------------------------------------------
# IPA → ARPAbet conversion
# ---------------------------------------------------------------------------

def _parse_ipa_token(token: str) -> List[str]:
    """
    Greedily parse one IPA token string into a list of ARPAbet phonemes.

    Examples
    --------
    'haɪ'  → ['HH', 'AY']
    'ðɪs'  → ['DH', 'IH', 'S']
    'aɪhæv'→ ['AY', 'HH', 'AE', 'V']
    'ŋ'    → ['NG']
    ''     → []
    """
    result = []
    s      = token.lower().strip()
    pos    = 0
    while pos < len(s):
        if s[pos] not in _IPA_START_CHARS:
            pos += 1
            continue
        matched = False
        for ipa_seq, arpa in _IPA_MAP_ORDERED:
            if s.startswith(ipa_seq, pos):
                result.append(arpa)
                pos += len(ipa_seq)
                matched = True
                break
        if not matched:
            pos += 1
    return result


def _token_to_arpabet(token: str) -> List[str]:
    """
    Convert one vocabulary token to ARPAbet phoneme(s).

    Tries three strategies in order:
      1. Direct ARPAbet match (uppercase label like 'HH', 'AY')
      2. Lowercase ARPAbet match (like 'hh', 'ay')
      3. IPA greedy parser (like 'h', 'aɪ', 'haɪðɪs')
    """
    if not token or token in _SKIP_TOKENS:
        return []

    # Strategy 1: direct uppercase ARPAbet
    upper = token.strip().upper()
    if upper in _VALID_ARPABET:
        return [upper]

    # Strategy 2: lowercase ARPAbet (e.g. model outputs 'hh', 'ay')
    lower_upper = token.strip().lower().upper()
    if lower_upper in _VALID_ARPABET:
        return [lower_upper]

    # Strategy 3: IPA greedy parse
    return _parse_ipa_token(token)


# ---------------------------------------------------------------------------
# CTC helpers
# ---------------------------------------------------------------------------

def _ctc_collapse(phonemes: List[str]) -> List[str]:
    """Remove consecutive duplicate phonemes (CTC blank-collapse artifact)."""
    if not phonemes:
        return []
    result = [phonemes[0]]
    for ph in phonemes[1:]:
        if ph != result[-1]:
            result.append(ph)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def recognize_phonemes(audio_path: str | Path) -> Optional[List[str]]:
    """
    Run wav2vec2 phoneme recognition on a (word-level) audio clip.

    Returns
    -------
    List[str]  – ARPAbet phonemes (may be [] for silence/noise)
    None       – model not available; caller should use fallback
    """
    try:
        return _run_inference(audio_path)
    except RuntimeError as exc:
        logger.error("Phoneme model unavailable: %s", exc)
        return None
    except Exception as exc:
        logger.exception("Phoneme recognition failed for '%s': %s", audio_path, exc)
        return None


def recognize_phonemes_for_word(
    audio_path: str | Path,
    word: str,
    expected_phonemes: List[str],
) -> Optional[List[str]]:
    result = recognize_phonemes(audio_path)
    if result is not None:
        logger.info("word='%s'  detected=%s  expected=%s", word, result, expected_phonemes)
    return result


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

def _run_inference(audio_path: str | Path) -> List[str]:
    import torch

    processor, model = _get_model()
    audio_np = _load_audio_as_float32(audio_path)

    # perform basic normalization & noise reduction to improve robustness in
    # noisy environments (user complaints indicated recognition fails when the
    # speaker is in a noisy area).
    try:
        from infrastructure.audio_processing import normalise, denoise

        audio_np = normalise(audio_np)
        audio_np = denoise(audio_np)
    except ImportError:
        logger.debug("could not import audio_processing helpers; skipping denoise")

    if len(audio_np) < SAMPLE_RATE * 0.08:   # < 80 ms – too short
        logger.warning("Audio clip too short (%d samples); returning []", len(audio_np))
        return []

    inputs = processor(
        audio_np,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )

    with torch.no_grad():
        logits = model(**inputs).logits   # (1, T, vocab)

    # ── Decode per-frame tokens (not batch_decode which concatenates) ──────
    predicted_ids = torch.argmax(logits, dim=-1)[0]   # (T,)
    vocab_tokens  = processor.tokenizer.convert_ids_to_tokens(
        predicted_ids.tolist()
    )

    blank = processor.tokenizer.pad_token   # typically '<pad>'

    # CTC collapse at token level: remove blank, collapse consecutive dups
    collapsed_tokens: List[str] = []
    prev = None
    for tok in vocab_tokens:
        if tok == blank:
            prev = None
            continue
        if tok in _SKIP_TOKENS:
            prev = None
            continue
        if tok == prev:
            continue
        collapsed_tokens.append(tok)
        prev = tok

    logger.debug("Collapsed tokens: %s", collapsed_tokens)

    # ── Map each token to ARPAbet ──────────────────────────────────────────
    arpabet: List[str] = []
    for tok in collapsed_tokens:
        arpa_list = _token_to_arpabet(tok)
        arpabet.extend(arpa_list)

    # Final collapse to remove any duplicates introduced by the IPA parser
    result = _ctc_collapse(arpabet)
    logger.info("phoneme result for '%s': tokens=%s → ARPAbet=%s",
                Path(audio_path).name, collapsed_tokens, result)
    return result
