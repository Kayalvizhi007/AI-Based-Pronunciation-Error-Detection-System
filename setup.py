#!/usr/bin/env python3
"""
setup.py — One-command project setup for Pronunciation Tutor
=============================================================

What this script does, in order:
  1. Checks Python version (needs 3.10 – 3.12)
  2. Creates a virtual environment in .venv/
  3. Upgrades pip inside the venv
  4. Installs faster-whisper (optimised Whisper backend)
  5. Installs all requirements from requirements.txt
  6. Downloads required NLTK data (cmudict)
  7. Creates .env from .env.example if it doesn't exist
  8. Creates the data/ directory for the SQLite database
  9. Optionally checks for ffmpeg (needed for WebM audio from browsers)
 10. Prints next steps

Run with:
    python setup.py
"""

import os
import sys
import subprocess
import shutil
import platform
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

HERE = Path(__file__).parent.resolve()
VENV = HERE / ".venv"

CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def c(colour, msg):
    return f"{colour}{msg}{RESET}"

def step(n, msg):
    print(f"\n{BOLD}{CYAN}[{n}]{RESET} {msg}")

def ok(msg):
    print(f"    {GREEN}✓{RESET} {msg}")

def warn(msg):
    print(f"    {YELLOW}⚠{RESET}  {msg}")

def fail(msg):
    print(f"    {RED}✗{RESET}  {msg}")

def run(cmd, check=True, capture=False):
    """Run a shell command and return CompletedProcess."""
    return subprocess.run(
        cmd,
        check=check,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        text=True,
    )

def venv_python():
    """Return path to Python inside the virtual environment."""
    if platform.system() == "Windows":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"

def venv_pip():
    """Return path to pip inside the virtual environment."""
    if platform.system() == "Windows":
        return VENV / "Scripts" / "pip.exe"
    return VENV / "bin" / "pip"

# ──────────────────────────────────────────────────────────────
# Step 1 – Python version check
# ──────────────────────────────────────────────────────────────

step(1, "Checking Python version")

major, minor = sys.version_info[:2]
if major != 3 or minor < 10 or minor > 12:
    fail(f"Python 3.10–3.12 required. You have {major}.{minor}.")
    fail("Download Python 3.11 from https://python.org/downloads")
    sys.exit(1)
ok(f"Python {major}.{minor} — compatible")

# ──────────────────────────────────────────────────────────────
# Step 2 – Create virtual environment
# ──────────────────────────────────────────────────────────────

step(2, "Creating virtual environment in .venv/")

if VENV.exists():
    warn(".venv/ already exists — skipping creation")
else:
    run([sys.executable, "-m", "venv", str(VENV)])
    ok("Virtual environment created")

# ──────────────────────────────────────────────────────────────
# Step 3 – Upgrade pip
# ──────────────────────────────────────────────────────────────

step(3, "Upgrading pip")
run([str(venv_python()), "-m", "pip", "install", "--upgrade", "pip", "--quiet"])
ok("pip is up to date")

# ──────────────────────────────────────────────────────────────
# Step 4 – Install faster-whisper first (separate so errors are clear)
# ──────────────────────────────────────────────────────────────

step(4, "Installing faster-whisper (optimised Whisper backend)")
print("    This may take a minute on first run...")
try:
    run([str(venv_pip()), "install", "faster-whisper>=1.0.0", "--quiet"])
    ok("faster-whisper installed")
except subprocess.CalledProcessError:
    warn("faster-whisper install failed — will fall back to openai-whisper")

# ──────────────────────────────────────────────────────────────
# Step 5 – Install all requirements
# ──────────────────────────────────────────────────────────────

step(5, "Installing project requirements from requirements.txt")
print("    This installs: Streamlit, PyTorch, transformers, Whisper, gTTS, etc.")
print("    First run may take 5–10 minutes depending on internet speed.")

req_file = HERE / "requirements.txt"
if not req_file.exists():
    fail("requirements.txt not found — are you running from the project root?")
    sys.exit(1)

try:
    run([str(venv_pip()), "install", "-r", str(req_file), "--quiet"])
    ok("All requirements installed")
except subprocess.CalledProcessError as e:
    fail("Some packages failed to install.")
    print("    Try running manually:")
    print(f"    {YELLOW}{venv_pip()} install -r requirements.txt{RESET}")
    sys.exit(1)

# ──────────────────────────────────────────────────────────────
# Step 6 – Download NLTK cmudict
# ──────────────────────────────────────────────────────────────

step(6, "Downloading NLTK cmudict (CMU Pronouncing Dictionary)")
try:
    result = run(
        [str(venv_python()), "-c",
         "import nltk; nltk.download('cmudict', quiet=True); "
         "from nltk.corpus import cmudict; d=cmudict.dict(); "
         "print(f'CMUdict loaded: {len(d)} words')"],
        capture=True,
    )
    ok(result.stdout.strip() or "cmudict ready")
except subprocess.CalledProcessError:
    warn("cmudict download failed — app will try again on first run")

# ──────────────────────────────────────────────────────────────
# Step 7 – Create .env file
# ──────────────────────────────────────────────────────────────

step(7, "Setting up environment configuration (.env)")

env_file     = HERE / ".env"
env_example  = HERE / ".env.example"

if env_file.exists():
    warn(".env already exists — not overwriting")
    warn("Edit it manually to set your GEMINI_API_KEY")
else:
    if env_example.exists():
        shutil.copy(env_example, env_file)
        ok(".env created from .env.example")
    else:
        # Create a minimal .env
        env_file.write_text(
            "GEMINI_API_KEY=your_gemini_api_key_here\n"
            "WHISPER_MODEL=base\n"
            "ASR_BACKEND=faster_whisper\n"
            "WHISPER_DEVICE=cpu\n"
            "WHISPER_COMPUTE_TYPE=int8\n"
            "LLM_BACKEND=auto\n"
            "OLLAMA_MODEL=qwen2.5:1.5b\n"
            "OLLAMA_BASE_URL=http://127.0.0.1:11434\n"
            "DB_PATH=data/pronunciation_tutor.db\n"
            "LOG_LEVEL=INFO\n",
            encoding="utf-8",
        )
        ok(".env created with default values")

print(f"\n    {YELLOW}ACTION REQUIRED:{RESET} Open .env and set your Gemini API key:")
print(f"    Get a free key at {CYAN}https://aistudio.google.com/app/apikey{RESET}")

# ──────────────────────────────────────────────────────────────
# Step 8 – Create data/ directory
# ──────────────────────────────────────────────────────────────

step(8, "Creating data/ directory for SQLite database")
(HERE / "data").mkdir(exist_ok=True)
ok("data/ directory ready")

# ──────────────────────────────────────────────────────────────
# Step 9 – Check for ffmpeg
# ──────────────────────────────────────────────────────────────

step(9, "Checking for ffmpeg (needed to handle browser WebM audio)")

if shutil.which("ffmpeg"):
    result = run(["ffmpeg", "-version"], capture=True, check=False)
    version_line = (result.stdout or "").split("\n")[0]
    ok(f"ffmpeg found — {version_line}")
else:
    warn("ffmpeg NOT found in PATH")
    warn("The app can still run without it (uses torchaudio fallback).")
    warn("For best audio compatibility install ffmpeg:")
    if platform.system() == "Windows":
        print(f"    {YELLOW}→ Windows:{RESET}  winget install ffmpeg")
        print(f"               or download from https://ffmpeg.org/download.html")
        print( "               and add the bin/ folder to your PATH")
    elif platform.system() == "Darwin":
        print(f"    {YELLOW}→ macOS:{RESET}    brew install ffmpeg")
    else:
        print(f"    {YELLOW}→ Linux:{RESET}    sudo apt install ffmpeg")

# ──────────────────────────────────────────────────────────────
# Step 10 – Optional service guidance
# ──────────────────────────────────────────────────────────────

step(10, "Optional external services")

print("""
    ┌─────────────────────────────────────────────────────────────┐
    │  OLLAMA  (local offline LLM — optional but recommended)     │
    │                                                             │
    │  1. Download from  https://ollama.com/download              │
    │  2. Install and run:                                        │
    │       ollama pull qwen2.5:1.5b                              │
    │       ollama serve                                          │
    │  3. Set in .env:  LLM_BACKEND=auto                         │
    │     (the app will automatically use Ollama when Gemini      │
    │      quota runs out)                                        │
    └─────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────┐
    │  MONTREAL FORCED ALIGNER  (optional — best accuracy)        │
    │                                                             │
    │  MFA gives the most precise phoneme timestamps but          │
    │  requires Conda.  WITHOUT it the system automatically       │
    │  uses Whisper timestamps + wav2vec2 (works well).           │
    │                                                             │
    │  To install MFA:                                            │
    │    conda create -n aligner -c conda-forge                  │
    │                  montreal-forced-aligner                    │
    │    conda activate aligner                                   │
    │    mfa model download acoustic english_us_arpa              │
    │    mfa model download dictionary english_us_arpa            │
    └─────────────────────────────────────────────────────────────┘
""")

# ──────────────────────────────────────────────────────────────
# Final instructions
# ──────────────────────────────────────────────────────────────

activate = (
    r".venv\Scripts\activate"
    if platform.system() == "Windows"
    else "source .venv/bin/activate"
)

print(f"""
{BOLD}{GREEN}╔══════════════════════════════════════════════════════╗
║           Setup complete!  Next steps:              ║
╚══════════════════════════════════════════════════════╝{RESET}

  1. Set your Gemini API key in .env
     {CYAN}GEMINI_API_KEY=your_key_here{RESET}

  2. Activate the virtual environment:
     {CYAN}{activate}{RESET}

  3. Launch the app:
     {CYAN}streamlit run app/streamlit_app.py{RESET}

  4. Open your browser at  {CYAN}http://localhost:8501{RESET}

  Optional diagnostic (run if phoneme scores look wrong):
     {CYAN}python diagnose_ipa.py{RESET}
""")
