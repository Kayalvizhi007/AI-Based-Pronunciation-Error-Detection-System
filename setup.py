#!/usr/bin/env python3
"""
One-command setup for Pronunciation Tutor.

Usage:
    python setup.py
    python setup.py --run
    python setup.py --dev
"""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent.resolve()
VENV = HERE / ".venv"

CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"


def c(color: str, message: str) -> str:
    return f"{color}{message}{RESET}"


def step(number: int, message: str) -> None:
    print(f"\n{BOLD}{CYAN}[{number}]{RESET} {message}")


def ok(message: str) -> None:
    print(f"    {GREEN}[ok]{RESET} {message}")


def warn(message: str) -> None:
    print(f"    {YELLOW}[warn]{RESET} {message}")


def fail(message: str) -> None:
    print(f"    {RED}[fail]{RESET} {message}")


def run(cmd: list[str], check: bool = True, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=check,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        text=True,
    )


def venv_python() -> Path:
    if platform.system() == "Windows":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def requirements_file(include_dev: bool) -> Path:
    return HERE / ("requirements-dev.txt" if include_dev else "requirements.txt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Set up Pronunciation Tutor locally.")
    parser.add_argument(
        "--run",
        action="store_true",
        help="Launch the Streamlit app after setup completes.",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Also install developer tools from requirements-dev.txt.",
    )
    return parser.parse_args()


def check_python_version() -> None:
    step(1, "Checking Python version")
    major, minor = sys.version_info[:2]
    if major != 3 or minor < 10 or minor > 12:
        fail(f"Python 3.10-3.12 required. You have {major}.{minor}.")
        fail("Download Python 3.11 from https://www.python.org/downloads/")
        sys.exit(1)
    ok(f"Python {major}.{minor} is compatible")


def ensure_venv() -> None:
    step(2, "Creating virtual environment in .venv/")
    if VENV.exists():
        ok(".venv already exists")
        return
    run([sys.executable, "-m", "venv", str(VENV)])
    ok("Virtual environment created")


def upgrade_pip() -> None:
    step(3, "Upgrading pip inside the virtual environment")
    run([str(venv_python()), "-m", "pip", "install", "--upgrade", "pip"])
    ok("pip upgraded")


def install_faster_whisper() -> None:
    step(4, "Installing faster-whisper")
    print("    This may take a minute on first run.")
    try:
        run([str(venv_python()), "-m", "pip", "install", "faster-whisper>=1.0.0"])
        ok("faster-whisper installed")
    except subprocess.CalledProcessError:
        warn("faster-whisper install failed; setup will continue with openai-whisper fallback")


def install_requirements(include_dev: bool) -> None:
    step(5, "Installing project dependencies")
    req_file = requirements_file(include_dev)
    if not req_file.exists():
        fail(f"{req_file.name} not found. Run setup from the project root.")
        sys.exit(1)

    print(f"    Using {req_file.name}")
    print("    First run may take several minutes because models and audio packages are large.")
    try:
        run([str(venv_python()), "-m", "pip", "install", "-r", str(req_file)])
        ok("Dependencies installed")
    except subprocess.CalledProcessError:
        fail("Dependency installation failed.")
        print(f"    Try again manually with: {c(YELLOW, f'{venv_python()} -m pip install -r {req_file.name}')}")
        sys.exit(1)


def download_cmudict() -> None:
    step(6, "Downloading NLTK cmudict")
    try:
        result = run(
            [
                str(venv_python()),
                "-c",
                (
                    "import nltk; "
                    "nltk.download('cmudict', quiet=True); "
                    "from nltk.corpus import cmudict; "
                    "print(f'CMUdict ready: {len(cmudict.dict())} words')"
                ),
            ],
            capture=True,
        )
        ok(result.stdout.strip() or "cmudict ready")
    except subprocess.CalledProcessError:
        warn("cmudict download failed; the app will retry on first run")


def _gemini_key_status() -> str:
    env_file = HERE / ".env"
    if not env_file.exists():
        return "missing"
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        if raw_line.startswith("GEMINI_API_KEY="):
            value = raw_line.split("=", 1)[1].strip()
            if value and value != "your_gemini_api_key_here":
                return "configured"
            return "placeholder"
    return "missing"


def ensure_env_file() -> None:
    step(7, "Preparing .env configuration")
    env_file = HERE / ".env"
    env_example = HERE / ".env.example"

    if env_file.exists():
        ok(".env already exists")
    elif env_example.exists():
        shutil.copy(env_example, env_file)
        ok(".env created from .env.example")
    else:
        env_file.write_text(
            "\n".join(
                [
                    "GEMINI_API_KEY=",
                    "WHISPER_MODEL=base",
                    "ASR_BACKEND=faster_whisper",
                    "WHISPER_DEVICE=cpu",
                    "WHISPER_COMPUTE_TYPE=int8",
                    "MFA_ACOUSTIC_MODEL=english_us_arpa",
                    "MFA_DICTIONARY=english_us_arpa",
                    "DB_PATH=data/pronunciation_tutor.db",
                    "GEMINI_MODEL=gemini-2.5-flash",
                    "LLM_BACKEND=auto",
                    "OLLAMA_MODEL=qwen2.5:1.5b",
                    "OLLAMA_BASE_URL=http://127.0.0.1:11434",
                    "OLLAMA_TIMEOUT_SEC=120",
                    "LOG_LEVEL=INFO",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        ok(".env created with default values")

    key_status = _gemini_key_status()
    if key_status == "configured":
        ok("Gemini API key detected in .env")
    else:
        warn("Gemini API key is not configured")
        print("    The app still works with Ollama or the built-in rule-based fallback.")
        print(f"    Optional key: {c(CYAN, 'https://aistudio.google.com/app/apikey')}")


def ensure_data_dir() -> None:
    step(8, "Creating data directory")
    (HERE / "data").mkdir(exist_ok=True)
    ok("data/ directory ready")


def check_ffmpeg() -> None:
    step(9, "Checking for ffmpeg")
    if shutil.which("ffmpeg"):
        result = run(["ffmpeg", "-version"], capture=True, check=False)
        version_line = (result.stdout or "").splitlines()[0]
        ok(f"ffmpeg found: {version_line}")
        return

    warn("ffmpeg not found in PATH")
    warn("The app can still run, but browser audio compatibility is better with ffmpeg installed.")
    if platform.system() == "Windows":
        print(f"    Windows: {c(YELLOW, 'winget install ffmpeg')}")
    elif platform.system() == "Darwin":
        print(f"    macOS:   {c(YELLOW, 'brew install ffmpeg')}")
    else:
        print(f"    Linux:   {c(YELLOW, 'sudo apt install ffmpeg')}")


def print_next_steps(args: argparse.Namespace) -> None:
    step(10, "Next steps")
    launch_line = f"{venv_python()} {HERE / 'run.py'}" if args.run else "python run.py"
    print(
        f"""
{BOLD}{GREEN}Setup complete.{RESET}

  1. Review {c(CYAN, '.env')} if you want Gemini or Ollama configuration changes.
  2. Start the app with {c(CYAN, launch_line)}
  3. Open {c(CYAN, 'http://localhost:8501')} in your browser
  4. Run {c(CYAN, 'python diagnose_ipa.py')} if phoneme output looks wrong

You can skip Gemini entirely. With {c(CYAN, 'LLM_BACKEND=auto')}, the app falls back to Ollama
or built-in explanations when no cloud key is available.
"""
    )


def maybe_launch_app(args: argparse.Namespace) -> None:
    if not args.run:
        return
    run_py = HERE / "run.py"
    if not run_py.exists():
        warn("run.py not found; skipping auto-launch")
        return

    step(11, "Launching the app")
    raise SystemExit(run([str(venv_python()), str(run_py)], check=False).returncode)


def main() -> None:
    args = parse_args()
    check_python_version()
    ensure_venv()
    upgrade_pip()
    install_faster_whisper()
    install_requirements(include_dev=args.dev)
    download_cmudict()
    ensure_env_file()
    ensure_data_dir()
    check_ffmpeg()
    print_next_steps(args)
    maybe_launch_app(args)


if __name__ == "__main__":
    main()
