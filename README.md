# Phoneme-Level Pronunciation Tutor

An AI-powered pronunciation coaching system that listens to your spoken English,
detects exactly which sounds you got wrong at the phoneme level, and guides you
to correct them — word by word.

---

## Quick Start

```bash
# 1. Run the automated setup script (creates venv, installs everything)
python setup.py

# 2. Copy and fill in your API key
copy .env.example .env
# Edit .env — set GEMINI_API_KEY=your_key_here

# 3. Activate the virtual environment
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux

# 4. Launch the app
streamlit run app/streamlit_app.py
```

Then open http://localhost:8501 in your browser.

---

## What the System Does

1. **You speak** a sentence into the microphone.
2. **Whisper ASR** transcribes what you said and records when each word was spoken.
3. **Gemini LLM** corrects any spelling errors in the transcript and asks you to confirm.
4. **wav2vec2** analyzes the audio of each individual word and extracts the phoneme sounds you actually produced.
5. **CMUdict** provides the correct phoneme sequence for every word.
6. The system **compares** expected vs. detected phonemes and finds substitutions, deletions, and insertions.
7. The **LLM Tutor** explains each mistake in plain English with mouth-position guidance.
8. You see a **word-by-word report** and an **overall session report** with scores, charts, and a personalised improvement plan.

---

## Project Layout

```
pronunciation_tutor/
├── app/
│   ├── streamlit_app.py          ← Main UI (record → confirm → report)
│   ├── analyzer.py               ← Orchestrates the full analysis pipeline
│   └── pages/
│       └── 2_Overall_Report.py   ← Full session report page
├── domain/
│   ├── error_detection.py        ← Detects substitution / deletion / insertion
│   ├── severity_scoring.py       ← Scores error severity from confidence
│   ├── learning_logic.py         ← Pass/fail threshold, retry logic
│   ├── phoneme_alignment.py      ← WordAlignment data structures
│   └── phoneme_scoring.py        ← Feature-weighted phoneme similarity
├── services/
│   ├── asr_service.py            ← Whisper speech-to-text + word timestamps
│   ├── mfa_service.py            ← Alignment pipeline (MFA → wav2vec2 → CMUdict)
│   ├── phoneme_recognition_service.py  ← wav2vec2 IPA→ARPAbet decoder
│   ├── llm_service.py            ← Gemini / Ollama / rule-based tutor
│   ├── tts_service.py            ← Text-to-speech (word playback)
│   └── tts_audio_service.py      ← Audio byte generation for browser
├── infrastructure/
│   ├── database.py               ← SQLite session & result storage
│   ├── audio_processing.py       ← Audio utility helpers
│   └── logging_config.py         ← Structured logging setup
├── assets/
│   └── articulation_images/      ← Mouth-position diagrams (optional)
├── data/
│   └── pronunciation_tutor.db    ← SQLite database (auto-created)
├── setup.py                      ← One-command automated setup
├── diagnose_ipa.py               ← Quick IPA parser self-test
├── requirements.txt
├── .env.example
└── README.md
```

---

## Configuration (.env)

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | *(required)* | Get free key at https://aistudio.google.com |
| `WHISPER_MODEL` | `base` | `tiny` fastest · `base` balanced · `small/medium` more accurate |
| `ASR_BACKEND` | `faster_whisper` | `faster_whisper` (recommended) or `openai_whisper` |
| `WHISPER_DEVICE` | `cpu` | `cpu` or `cuda` (if you have an NVIDIA GPU) |
| `LLM_BACKEND` | `auto` | `auto` tries Gemini then Ollama then rule-based |
| `OLLAMA_MODEL` | `qwen2.5:1.5b` | Local LLM model name (if Ollama is installed) |
| `DB_PATH` | `data/pronunciation_tutor.db` | SQLite file path |

---

## External Services

### Gemini API (required for LLM tutor)
1. Go to https://aistudio.google.com/app/apikey
2. Create a free API key
3. Paste it in `.env` as `GEMINI_API_KEY=...`

The free tier gives 15 requests/minute and 1,000 requests/day — enough for testing.
If Gemini quota runs out, the system automatically falls back to rule-based explanations.

### Ollama (optional — offline LLM fallback)
Install from https://ollama.com/download then run:
```bash
ollama pull qwen2.5:1.5b
ollama serve
```
Ollama runs at http://localhost:11434. Set `LLM_BACKEND=ollama` in `.env` to prefer it.

### Montreal Forced Aligner (optional — best phoneme accuracy)
MFA gives the most precise word/phoneme timestamps but requires Conda.
Without it the system uses Whisper timestamps + wav2vec2, which works well.

```bash
conda create -n aligner -c conda-forge montreal-forced-aligner
conda activate aligner
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa
```

---

## Scoring System

Phoneme accuracy uses **feature-weighted similarity** rather than binary matching.
Each ARPAbet phoneme is described by articulatory features (place, manner, voicing,
vowel height, backness, tenseness). Two phonemes are compared by Jaccard overlap
of their feature sets:

| Example | Expected | Detected | Old score | New score |
|---|---|---|---|---|
| "think" said as "tink" | TH IH NG K | T IH NG K | 75% | 82% |
| "these" said as "diz" | DH IY Z | D IH Z | 33% | 60% |
| "is" said as "iz" | IH Z | IH Z | 100% | 100% |

Pass threshold: **70%** feature-weighted accuracy per word.

---

## Diagnostics

If phoneme detection seems wrong, run the IPA parser self-test:
```bash
python diagnose_ipa.py
```
Expected output: all green ticks. This confirms the wav2vec2 → ARPAbet conversion is working.

---

## Requirements

- Python 3.10 – 3.12
- Windows 10/11, macOS 12+, or Ubuntu 20.04+
- Microphone access in browser
- ~2 GB disk space (Whisper base model + wav2vec2)
- Internet connection for first run (model downloads)

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: faster_whisper` | Run `pip install faster-whisper` inside the venv |
| All words score 0/100 | Run `python diagnose_ipa.py` to check IPA parser |
| Gemini 429 error | Free quota (20 req/day for flash); wait or use Ollama |
| No audio detected | Allow microphone in browser; use Chrome/Edge |
| MFA not found | Expected — system uses wav2vec2 fallback automatically |
| `pydub` AudioSegment error | Install ffmpeg: `winget install ffmpeg` or from https://ffmpeg.org |
