#!/usr/bin/env python3
"""
Launch the Streamlit app with the project's virtual environment when available.

Usage:
    python run.py
    python run.py --check
    python run.py --server.port 8502
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent.resolve()
APP_FILE = HERE / "app" / "streamlit_app.py"
ENV_FILE = HERE / ".env"
ENV_EXAMPLE = HERE / ".env.example"
VENV = HERE / ".venv"


def venv_python() -> Path:
    if os.name == "nt":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def using_virtualenv() -> bool:
    return bool(os.getenv("VIRTUAL_ENV")) or sys.prefix != getattr(sys, "base_prefix", sys.prefix)


def reexec_in_venv(args: list[str]) -> int:
    candidate = venv_python()
    if using_virtualenv() or not candidate.exists():
        return -1
    print(f"Using virtual environment: {candidate}", flush=True)
    return subprocess.call([str(candidate), str(HERE / "run.py"), *args])


def ensure_env_file() -> None:
    if ENV_FILE.exists() or not ENV_EXAMPLE.exists():
        return
    shutil.copy(ENV_EXAMPLE, ENV_FILE)
    print("Created .env from .env.example")


def gemini_key_status() -> str:
    if not ENV_FILE.exists():
        return "missing"
    for raw_line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        if raw_line.startswith("GEMINI_API_KEY="):
            value = raw_line.split("=", 1)[1].strip()
            if value and value != "your_gemini_api_key_here":
                return "configured"
            return "placeholder"
    return "missing"


def print_check() -> int:
    ensure_env_file()
    status = gemini_key_status()
    print(f"Python: {sys.executable}")
    print(f"App file: {'ok' if APP_FILE.exists() else 'missing'}")
    print(f".env: {'ok' if ENV_FILE.exists() else 'missing'}")
    print(f"Gemini key: {status}")
    print(f"ffmpeg: {'found' if shutil.which('ffmpeg') else 'missing'}")
    return 0 if APP_FILE.exists() else 1


def launch(streamlit_args: list[str]) -> int:
    ensure_env_file()

    if not APP_FILE.exists():
        print("App entrypoint not found. Expected app/streamlit_app.py.")
        return 1

    try:
        import streamlit  # noqa: F401
    except ImportError:
        print("Streamlit is not installed in this interpreter.")
        print("Run `python setup.py` first.")
        return 1

    if gemini_key_status() != "configured":
        print("Gemini API key not configured. The app will use Ollama or rule-based fallback.")

    cmd = [sys.executable, "-m", "streamlit", "run", str(APP_FILE), *streamlit_args]
    return subprocess.call(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate local setup and exit.",
    )
    args, streamlit_args = parser.parse_known_args()

    reexec_code = reexec_in_venv(sys.argv[1:])
    if reexec_code >= 0:
        raise SystemExit(reexec_code)

    if args.check:
        raise SystemExit(print_check())
    raise SystemExit(launch(streamlit_args))


if __name__ == "__main__":
    main()
