# Phoneme-Level Pronunciation Tutor

An AI pronunciation coach that records spoken English, detects phoneme-level mistakes, and gives word-by-word feedback in a Streamlit app.

## Quick Start

```bash
git clone https://github.com/AadhithyaPrakash/pronunciation_tutor.git
cd pronunciation_tutor
python setup.py
python run.py
```

Then open `http://localhost:8501`.

`setup.py` creates `.venv`, installs dependencies, downloads `cmudict`, prepares `.env`, and creates the SQLite data directory.

## Easiest Usage Flow

1. Run `python setup.py` once.
2. Optionally edit `.env` if you want:
   - `GEMINI_API_KEY` for Gemini-backed transcript correction and coaching
   - `LLM_BACKEND=ollama` for a local LLM
3. Start the app with `python run.py`
4. Record a sentence in the browser and review the word-level report
> **Noise handling**: the backend now applies basic noise reduction and peak
> normalization to recorded audio before both transcription and phoneme
> decoding.  This helps the system focus on the user’s speech even when the
> recording contains background hiss.  Despite the preprocessing, very loud
> or overlapping noises will still degrade accuracy – using a quieter room or
> a headset microphone is the most reliable fix.
You do not need to copy `.env.example` manually. The setup and launcher scripts handle that.

## Useful Commands

```bash
python setup.py --run      # setup, then launch immediately
python setup.py --dev      # include pytest for local development
python run.py --check      # quick local setup check
python diagnose_ipa.py     # verify IPA -> ARPAbet parsing
```

## What the App Does

1. Shows a **login / registration screen** before you start.
2. Records speech through your browser (microphone access required).
3. Transcribes the audio with Whisper and displays the detected text.
4. Lets the learner edit the detected text before running pronunciation analysis.
5. Aligns words and phonemes using timestamps + optional MFA alignment.
6. Compares expected and detected phonemes per word and generates:
   - Per-word phoneme comparison (expected vs detected)
   - Clear feedback & improvement tips
   - A score with emoji-based feedback and a final report

## Configuration

`.env` is optional for the basic app flow.

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | blank | Optional Gemini API key |
| `WHISPER_MODEL` | `base` | Whisper model size |
| `ASR_BACKEND` | `faster_whisper` | `faster_whisper` or `openai_whisper` |
| `WHISPER_DEVICE` | `cpu` | `cpu` or `cuda` |
| `LLM_BACKEND` | `auto` | Try Gemini, then Ollama, then rule-based fallback |
| `OLLAMA_MODEL` | `qwen2.5:1.5b` | Local Ollama model name |
| `DB_PATH` | `data/pronunciation_tutor.db` | SQLite database path |

If `GEMINI_API_KEY` is not set, the app still runs. Transcript correction falls back to a heuristic and tutor messaging falls back to Ollama or built-in rules.

## Optional Services

### Gemini

Create an API key at `https://aistudio.google.com/app/apikey` and place it in `.env` as `GEMINI_API_KEY=...`.

### Ollama

```bash
ollama pull qwen2.5:1.5b
ollama serve
```

Set `LLM_BACKEND=ollama` in `.env` if you want to force the local model.

### Montreal Forced Aligner

MFA gives the best phoneme alignment but is optional.

```bash
conda create -n aligner -c conda-forge montreal-forced-aligner
conda activate aligner
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa
```

## Requirements

- Python 3.10 to 3.12
- Windows 10/11, macOS 12+, or Ubuntu 20.04+
- Browser microphone access
- Internet for first-run model downloads
- Roughly 2 GB of disk space for models

`ffmpeg` is recommended for better browser audio compatibility but not strictly required.

## Troubleshooting

| Problem | Fix |
|---|---|
| `Streamlit is not installed in this interpreter` | Run `python setup.py` again, then use `python run.py` |
| `ModuleNotFoundError: faster_whisper` | Re-run `python setup.py` or install inside `.venv` |
| All words score `0/100` | Run `python diagnose_ipa.py` |
| Gemini quota or auth error | Set up Ollama or rely on rule-based fallback |
| No microphone audio | Allow browser microphone access and retry in Chrome or Edge |
| `ffmpeg` missing | Install with `winget install ffmpeg`, `brew install ffmpeg`, or your package manager |

## Development

```bash
python setup.py --dev
.venv\Scripts\python -m pytest tests      # Windows
.venv/bin/python -m pytest tests          # macOS / Linux
```
