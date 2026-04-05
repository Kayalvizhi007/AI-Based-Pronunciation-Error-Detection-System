"""
Pronunciation Checker – Main Page
-----------------------------------
Run with:
    streamlit run app/streamlit_app.py

Flow:
    1. Record audio
    2. Confirm / edit transcript
    3. Word-by-word phoneme analysis (this page)
    4. "View Full Report" → pages/2_Overall_Report.py
"""

from __future__ import annotations

import logging
import re
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import bcrypt
from infrastructure import database

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import streamlit as st


def validate_username(username: str) -> str | None:
    if len(username) != 6:
        return "Username must be exactly 6 characters: 4 letters followed by 2 numbers."
    if not re.match(r"[a-z]{4}\d{2}", username):
        return "Username must consist of exactly 4 lowercase letters followed by 2 numbers."
    return None


def validate_password(password: str) -> str | None:
    if len(password) != 7:
        return "Password must be exactly 7 characters: 4 letters, 1 symbol, 2 numbers."
    if not re.match(r"[a-z]{4}[^a-zA-Z0-9]{1}\d{2}", password):
        return "Password must consist of exactly 4 lowercase letters, 1 symbol, then 2 numbers."
    return None


try:
    from infrastructure.logging_config import configure_logging
    configure_logging()
except Exception:
    logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

from app.analyzer import PronunciationAnalyzer, PronunciationReport, WordReport
from services import tts_audio_service

import pandas as pd

st.set_page_config(
    page_title="Pronunciation Checker",
    page_icon="🗣️",
    layout="wide",
)

st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #f3f5ff 0%, #eef9ff 100%);
}
.stButton>button {
    background-color: #6c63ff;
    color: #fff;
    border: 0;
}
.stButton>button:hover {
    background-color: #5f55e8;
}
.stTextInput>div>div>input {
    border-color: #a5b4fc;
}
.block-container {
    padding: 1rem 2rem 2rem;
}
</style>
""", unsafe_allow_html=True)


def generate_pdf_bytes(text: str) -> bytes:
    """Generate a basic PDF from text, with fallback to raw bytes if fpdf is unavailable."""
    try:
        from fpdf import FPDF
    except ImportError:
        return text.encode("utf-8")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)

    for line in text.splitlines():
        pdf.multi_cell(0, 8, line)

    return pdf.output(dest="S").encode("latin-1")

st.markdown("""
<style>
.phoneme-tag {
    display: inline-block;
    padding: 3px 8px;
    border-radius: 5px;
    margin: 2px;
    font-family: monospace;
    font-size: 0.85rem;
    font-weight: 600;
}
.ph-correct  { background: #d5f5e3; color: #1a7a45; }
.ph-error    { background: #fde8e8; color: #c0392b; }
.score-pill {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-weight: 800;
    font-size: 1.1rem;
}
</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────

def _reset():
    st.session_state.analyzer        = PronunciationAnalyzer()
    st.session_state.raw_transcript   = ""
    st.session_state.corrected        = ""
    st.session_state.report           = None
    st.session_state.audio_path       = None
    st.session_state.stage            = "login"
    st.session_state.user             = None
    st.session_state.login_success    = False
    st.session_state.logged_in        = False
    st.session_state.username         = ""


def _reset_recording():
    st.session_state.analyzer        = PronunciationAnalyzer()
    st.session_state.raw_transcript   = ""
    st.session_state.corrected        = ""
    st.session_state.report           = None
    st.session_state.audio_path       = None
    st.session_state.stage            = "record"
    # Keep login state


def _render_navbar():
    if st.session_state.stage in ["login", "register"]:
        return

    user = st.session_state.user or {}
    username = user.get("username", "Guest")

    col_left, col_right = st.columns([4, 1])
    with col_left:
        st.markdown(f"**👋 Logged in as {username}**")
    with col_right:
        if st.button("Logout", key="logout", use_container_width=True):
            _reset()
            st.session_state.stage = "login"
            st.success("You have been logged out.")
            st.rerun()


if "stage" not in st.session_state:
    _reset()


_render_navbar()


def _save_audio(uploaded) -> Path:
    """Save uploaded audio bytes to a clean 16 kHz mono WAV file.

    Streamlit may deliver audio in different sample rates or stereo; this
    ensures we standardize the format for consistent Whisper transcription.
    """
    data = uploaded.read()

    try:
        import io
        import numpy as np
        import soundfile as sf
        from scipy.signal import resample

        buf = io.BytesIO(data)
        audio, sr = sf.read(buf, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Resample to 16 kHz if needed
        target_sr = 16000
        if sr != target_sr:
            num = int(len(audio) * target_sr / sr)
            audio = resample(audio, num)
            sr = target_sr

        # Normalize loudness to avoid clipping / low volume
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        sf.write(str(tmp.name), audio, sr, subtype="PCM_16")
        return Path(tmp.name)
    except Exception:
        # Fallback to raw bytes if conversion fails
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(data)
        tmp.close()
        return Path(tmp.name)


analyzer: PronunciationAnalyzer = st.session_state.analyzer

# Check login state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

if not st.session_state.logged_in and st.session_state.stage not in ["login", "register"]:
    st.session_state.stage = "login"
    st.rerun()

# ── Header ─────────────────────────────────────────────────────────────────
if st.session_state.stage in ["login", "register"]:
    st.title("🗣️ Pronunciation Checker - Login")
    st.caption("Please log in or register to continue.")
else:
    st.title("🗣️ Pronunciation Checker")
    st.caption("Record → get instant phoneme-level feedback on every word.")
st.divider()

# ==========================================================================
# STAGE 1 – LOGIN
# ==========================================================================
if st.session_state.stage == "login":
    st.subheader("Login")
    show_password = st.checkbox("Show password", key="login_show_password")
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="text" if show_password else "password", key="login_password")

    login_message = None
    if st.button("Login", type="primary"):
        if not username or not password:
            login_message = ("error", "Please enter both username and password.")
        else:
            user = database.verify_user(username, password)
            if not user:
                login_message = ("error", "Invalid Username or Password")
            else:
                st.session_state.user = user
                st.session_state.stage = "main"
                st.session_state.login_success = True
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("Login Successful")
                st.rerun()

    if login_message:
        st.error(login_message[1])

    if st.button("New User? Register"):
        st.session_state.stage = "register"
        st.rerun()

# ==========================================================================
# STAGE 2 – REGISTER
# ==========================================================================
elif st.session_state.stage == "register":
    st.subheader("Register")
    show_password = st.checkbox("Show passwords", key="reg_show_password")
    username = st.text_input("Username", key="reg_username")
    email = st.text_input("Email", key="reg_email")
    password = st.text_input("Password", type="text" if show_password else "password", key="reg_password")
    confirm_password = st.text_input("Confirm Password", type="text" if show_password else "password", key="reg_confirm_password")

    username_error = validate_username(username) if username else "Username must be exactly 6 characters: 4 lowercase letters followed by 2 numbers."
    password_error = validate_password(password) if password else "Password must be exactly 7 characters: 4 lowercase letters, 1 symbol, then 2 numbers."
    confirm_error = None
    email_error = None
    if email and not re.match(r"[^@]+@[^@]+\.[^@]+", email):
        email_error = "Please enter a valid email address."
    if confirm_password and password != confirm_password:
        confirm_error = "Passwords do not match."

    if username:
        if username_error:
            st.warning(username_error)
        else:
            st.success("Username format is valid.")

    if email:
        if email_error:
            st.warning(email_error)
        else:
            st.success("Email format is valid.")

    if password:
        if password_error:
            st.warning(password_error)
        else:
            st.success("Password format is valid.")

    if confirm_password:
        if confirm_error:
            st.warning(confirm_error)
        else:
            st.success("Passwords match.")

    if st.button("Register", type="primary"):
        if username_error or password_error or confirm_error or email_error:
            st.error("Fix validation errors before registering.")
        elif database.get_user_by_username(username) is not None:
            st.error("Username already exists.")
        else:
            password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            user_id = database.create_user(username, email, password_hash)
            user = database.get_user(user_id)
            st.session_state.user = user
            st.session_state.stage = "main"
            st.session_state.login_success = True
            st.success("Registration Successful")
            st.rerun()

    if st.button("Back to Login"):
        st.session_state.stage = "login"
        st.rerun()

# ==========================================================================
# STAGE 3 – MAIN MENU
# ==========================================================================
elif st.session_state.stage == "main":
    user = st.session_state.user or {}
    st.subheader(f"Welcome, {user.get('username', 'User')}!")
    if st.session_state.get("login_success", False):
        st.success("Login Successful")
        st.session_state.login_success = False

    c1, c2 = st.columns(2)
    with c1:
        if st.button("👤 Profile Viewer", use_container_width=True):
            st.session_state.stage = "profile"
            st.rerun()
        if st.button("🎙️ Record Speech", use_container_width=True):
            st.session_state.stage = "record"
            st.rerun()
    with c2:
        if st.button("📊 User Status", use_container_width=True):
            st.session_state.stage = "status"
            st.rerun()
        if st.button("📈 Score Details", use_container_width=True):
            st.session_state.stage = "score_details"
            st.rerun()

# ==========================================================================
# STAGE 3a – PROFILE VIEWER
# ==========================================================================
elif st.session_state.stage == "profile":
    st.subheader("👤 User Profile")
    user = st.session_state.user or {}
    db_user = database.get_user(user.get("id")) if user else None
    if not db_user:
        st.error("User details not found. Please log in again.")
        if st.button("Return to login"):
            _reset(); st.session_state.stage = "login"; st.rerun()
    else:
        sessions = database.get_user_sessions(db_user["id"], limit=100)
        total_sessions = len(sessions)
        total_sentences = total_sessions
        daily_stats = database.get_user_daily_stats(db_user["id"], days=30)
        practice_minutes = sum(r.get("practice_minutes", 0) for r in daily_stats) if daily_stats else 0

        avg_score = round(sum(s.get("score", 0) for s in sessions) / (total_sessions or 1))
        if avg_score >= 90:
            level = "Expert"
        elif avg_score >= 75:
            level = "Advanced"
        elif avg_score >= 60:
            level = "Intermediate"
        else:
            level = "Beginner"

        keys = [s.get("started_at", "").split("T")[0] for s in sessions if s.get("started_at")]
        streak = 0
        if keys:
            from datetime import date, timedelta
            dayset = set(keys)
            today = date.today()
            while True:
                date_key = (today - timedelta(days=streak)).isoformat()
                if date_key in dayset:
                    streak += 1
                else:
                    break

        practice_times = [int(s.get("started_at", "T00").split("T")[1][:2] if "T" in s.get("started_at", "") else 0) for s in sessions if s.get("started_at")]
        morning = sum(1 for h in practice_times if 5 <= h < 12)
        evening = sum(1 for h in practice_times if 17 <= h < 22)
        favorite_time = "Morning" if morning >= evening and morning > 0 else "Evening" if evening > 0 else "Anytime"

        completion = min(100, 40 + int(avg_score * 0.4) + min(total_sessions, 20))

        st.markdown(f"**Username:** {db_user['username']}  \n**Email:** {db_user['email']}  \n**Created:** {db_user.get('created_at', 'N/A')}  \n**Level:** {level}")
        st.metric("Total sessions", total_sessions)
        st.metric("Total sentences", total_sentences)
        st.metric("Current streak", streak)
        st.metric("Favorite practice time", favorite_time)
        st.metric("Practice minutes (30d)", practice_minutes)
        st.progress(completion / 100.0)
        st.caption(f"Profile completion: {completion}%")

        if sessions:
            st.subheader("Recent sessions")
            st.table([{"Date": s.get("started_at", ""), "Sentence": s.get("sentence", ""), "Score": s.get("score", 0)} for s in sessions[:10]])

    if st.button("Back to main"):
        st.session_state.stage = "main"
        st.rerun()

# ==========================================================================
# STAGE 3b – USER STATUS DASHBOARD
# ==========================================================================
elif st.session_state.stage == "status":
    st.subheader("📊 User Status Dashboard")
    user = st.session_state.user or {}
    db_user = database.get_user(user.get("id")) if user else None
    if not db_user:
        st.error("User details not found. Please log in again.")
        if st.button("Return to login"):
            _reset(); st.session_state.stage = "login"; st.rerun()
    else:
        history = database.get_user_session_counts(db_user["id"], days=30)

        import pandas as pd
        if history:
            df = pd.DataFrame(history)
            df["d"] = pd.to_datetime(df["d"])
            df = df.set_index("d")

            chart_type = st.selectbox(
                "Choose chart type for metrics:",
                ["bar", "line", "area"],
                index=0,
            )
            chart_fn = {
                "bar": st.bar_chart,
                "line": st.line_chart,
                "area": st.area_chart,
            }[chart_type]

            st.subheader("Weekly activity (sessions)")
            chart_fn(df["sessions"])
            st.subheader("Score trend")
            chart_fn(df["avg_score"])
        else:
            st.info("No activity data yet.")

        sessions = database.get_user_sessions(db_user["id"], limit=100)
        best_streak = max((s["total_sessions"] for s in database.get_user_daily_stats(db_user["id"], days=365)), default=0)
        days = sorted({s["started_at"][:10] for s in sessions if s.get("started_at")})
        streak = 0
        from datetime import date, timedelta
        dayset = set(days)
        today = date.today()
        while True:
            dkey = (today - timedelta(days=streak)).isoformat()
            if dkey in dayset:
                streak += 1
            else:
                break
        current_streak = streak

        total_time = sum(r.get("practice_minutes", 0) for r in database.get_user_daily_stats(db_user["id"], days=365))

        this_week = sum(s.get("score", 0) for s in sessions[:7]) / max(1, len(sessions[:7])) if sessions else 0
        last_week = sum(s.get("score", 0) for s in sessions[7:14]) / max(1, len(sessions[7:14])) if len(sessions) > 7 else 0
        diff = this_week - last_week

        st.metric("Current streak", current_streak)
        st.metric("Best streak", best_streak)
        st.metric("Total time practiced (min)", total_time)
        st.metric("This week vs last week", f"{this_week:.1f}% vs {last_week:.1f}%")
        st.markdown(f"**Predicted next week:** {max(0, min(100, this_week + diff * 0.5)):.1f}%")
        st.info("You improved {:+.1f}% this week! Keep going!".format(diff) if diff >= 0 else "Keep focus: every session counts")

    if st.button("Back to main"):
        st.session_state.stage = "main"
        st.rerun()

# ==========================================================================
# STAGE 3c – SCORE DETAILS
# ==========================================================================
elif st.session_state.stage == "score_details":
    st.subheader("🏅 Score Details")
    user = st.session_state.user or {}
    db_user = database.get_user(user.get("id")) if user else None
    if not db_user:
        st.error("User details not found. Please log in again.")
        if st.button("Return to login"):
            _reset(); st.session_state.stage = "login"; st.rerun()
    else:
        sessions = database.get_user_sessions(db_user["id"], limit=50)
        if not sessions:
            st.info("No sessions available yet.")
        else:
            latest = sessions[0]
            overall_score = latest.get("score", 0)
            if overall_score >= 91:
                label = "🎉 Excellent!"
            elif overall_score >= 76:
                label = "😊 Great Work"
            elif overall_score >= 61:
                label = "🙂 Good Job"
            elif overall_score >= 41:
                label = "😐 Getting Better"
            else:
                label = "😟 Needs Practice"

            st.markdown(f"### {overall_score} {label}")
            st.metric("Best score ever", f"{max(s.get('score',0) for s in sessions)} 🏆")
            if len(sessions) > 1:
                trend = "↑ Improving" if sessions[0]['score'] >= sessions[1]['score'] else "↓ Declining"
            else:
                trend = "—"
            st.metric("Latest trend", trend)

            st.subheader("Past scores")
            st.dataframe(pd.DataFrame([{"Date": s['started_at'], "Sentence": s['sentence'], "Score": s['score']} for s in sessions]))

            st.subheader("Progress by category")
            cat = {"Clarity": 80, "Fluency": 75, "Accuracy": 82, "Speed": 68}
            for k,v in cat.items():
                st.write(f"{k}: {v}%")
                st.progress(v / 100)

            report_text = "\n".join([f"{s['started_at']} | {s['sentence']} | {s['score']}" for s in sessions])
            report_bytes = generate_pdf_bytes(report_text)
            st.download_button("Download score report as PDF", report_bytes, file_name='score_report.pdf', mime='application/pdf')
            if st.button("Share score"):
                st.success("Link copied! (Simulation)")

    if st.button("Back to main"):
        st.session_state.stage = "main"
        st.rerun()

# ==========================================================================
# STAGE 4 – RECORD (existing)
# ==========================================================================
elif st.session_state.stage == "record":
    st.subheader("Step 1 — Record your sentence")
    st.info("Prepare your microphone and keep background noise low.")

    uploaded = st.file_uploader("Or upload an existing recording (WAV/MP3)", type=["wav", "mp3"], key="upload_audio")
    if uploaded and not st.session_state.audio_path:
        audio_path = _save_audio(uploaded)
        st.session_state.audio_path = audio_path
        raw = analyzer.transcribe(audio_path)
        if not raw or raw.strip() in ["", "—"]:
            st.error("No speech detected in upload. Please try another file or use the recorder.")
        else:
            st.session_state.raw_transcript = raw
            st.session_state.corrected = analyzer.correct_transcript(raw)
            st.session_state.stage = "confirm"
            st.rerun()

    if "recording_active" not in st.session_state:
        st.session_state.recording_active = False
    if "recording_start_time" not in st.session_state:
        st.session_state.recording_start_time = None

    if not st.session_state.recording_active:
        if st.button("🎙️ Start Recording", key="start_record"):
            st.session_state.recording_active = True
            st.session_state.recording_start_time = time.time()
            st.rerun()
    else:
        elapsed = int(time.time() - st.session_state.recording_start_time)
        minutes = elapsed // 60
        seconds = elapsed % 60
        st.markdown(f"**Recording Time: {minutes}:{seconds:02d}**")
        
        if st.button("⏹️ Stop Recording", key="stop_record"):
            st.session_state.recording_active = False
            duration = time.time() - st.session_state.recording_start_time
            if duration < 0.5:
                st.error("Recording too short. Please try again.")
                st.session_state.stage = "record"
                st.rerun()

            with st.spinner("Recording and processing..."):
                from infrastructure.audio_processing import record_to_tempfile, SAMPLE_RATE
                import numpy as np
                from pathlib import Path

                audio_path = record_to_tempfile(duration=int(duration) + 1)  # add 1 second buffer
                if not Path(audio_path).exists() or Path(audio_path).stat().st_size == 0:
                    st.error("Recorded audio file missing or empty. Please check your microphone and re-record.")
                    st.session_state.stage = "record"
                    st.rerun()

                st.session_state.audio_path = audio_path

                _audio = _save_audio(open(audio_path, 'rb'))
                st.session_state.audio_path = _audio

                raw = analyzer.transcribe(_audio)
                st.write(f"Debug: audio_path={_audio}, exists={Path(_audio).exists()}, size={Path(_audio).stat().st_size}")
                if not raw or raw.strip() in ["", "—"]:
                    st.error("No speech detected. Please re-record.")
                    st.write("Detected raw transcript:", repr(raw))
                    st.session_state.stage = "record"
                    st.rerun()

                st.session_state.raw_transcript = raw
                st.session_state.corrected = analyzer.correct_transcript(raw)
                st.session_state.stage = "confirm"
                st.rerun()
        else:
            # Continue updating timer
            time.sleep(0.1)
            st.rerun()

# ==========================================================================
# STAGE 2 – CONFIRM
# ==========================================================================
elif st.session_state.stage == "confirm":
    st.subheader("Step 2 — Confirm what you said")

    # Provide audio playback so the user can verify the recording
    if st.session_state.audio_path:
        st.markdown("**Playback the recording:**")
        with open(st.session_state.audio_path, "rb") as f:
            st.audio(f.read(), format="audio/wav")
        st.divider()

    col_raw, col_edit = st.columns(2)
    with col_raw:
        st.markdown("**🎤 We heard:**")
        st.code(st.session_state.raw_transcript or "—", language=None)
    with col_edit:
        st.markdown("**✏️ Edit if needed:**")
        edited = st.text_input(
            "Corrected sentence",
            value=st.session_state.corrected,
            label_visibility="collapsed",
            key="edit_sentence",
        )
        st.session_state.corrected = edited

    st.markdown("")
    col_a, col_r, _ = st.columns([2, 1, 4])
    with col_a:
        if st.button("🔍 Analyse Pronunciation", type="primary", use_container_width=True):
            with st.spinner("Analysing phonemes…"):
                try:
                    user = st.session_state.user or {}
                    user_id = user.get("id")
                    report = analyzer.analyze(
                        audio_path=st.session_state.audio_path,
                        sentence=st.session_state.corrected,
                        user_id=user_id,
                    )
                    st.session_state.report = report
                    st.session_state.stage  = "report"
                    st.rerun()
                except Exception as exc:
                    logger.exception("Analysis failed")
                    st.error(f"Analysis failed: {exc}")
    with col_r:
        if st.button("🔄 Re-record", use_container_width=True):
            _reset_recording()
            st.rerun()

# ==========================================================================
# STAGE 3 – WORD-BY-WORD ANALYSIS
# ==========================================================================
elif st.session_state.stage == "report":
    report: PronunciationReport = st.session_state.report
    score = report.overall_score

    # ── Mini score banner ──────────────────────────────────────────────────
    score_color = (
        "#27ae60" if score >= 75
        else "#f39c12" if score >= 50
        else "#e74c3c"
    )
    score_emoji = (
        "🌟" if score >= 90 else
        "✅" if score >= 75 else
        "👍" if score >= 50 else
        "💪"
    )

    banner_col, btn_col = st.columns([3, 1])
    with banner_col:
        st.markdown(
            f'<span class="score-pill" style="background:{score_color}22;'
            f'color:{score_color};border:2px solid {score_color}">'
            f'{score_emoji} Overall Score: {score} / 100</span>',
            unsafe_allow_html=True,
        )
        st.markdown(f'**Sentence:** *"{report.sentence}"*')
    with btn_col:
        if st.button("📊 View Full Report →", type="primary",
                     use_container_width=True, key="report_btn_top"):
            # Streamlit multisession expects a file path relative to the app directory
            st.switch_page("pages/2_Overall_Report.py")

    st.divider()

    # ── Word-by-word breakdown ─────────────────────────────────────────────
    st.subheader("📝 Word-by-Word Phoneme Analysis")

    for wr in report.word_reports:
        score_icon = "✅" if wr.score >= 80 else ("⚠️" if wr.score >= 50 else "❌")

        with st.expander(
            f"{score_icon}  **{wr.word.upper()}**  ·  {wr.score}/100",
            expanded=wr.has_errors,
        ):
            ph_col, det_col, audio_col = st.columns([3, 3, 2])

            # Expected phonemes
            with ph_col:
                st.markdown("**Expected:**")
                error_exp = {e["expected_phoneme"] for e in wr.errors if e.get("expected_phoneme")}
                tags = [
                    f'<span class="phoneme-tag {"ph-error" if ph in error_exp else "ph-correct"}">'
                    f'{ph}</span>'
                    for ph in wr.expected_phonemes
                ]
                st.markdown(" ".join(tags) or "*—*", unsafe_allow_html=True)

            # Detected phonemes
            with det_col:
                st.markdown("**You produced:**")
                if wr.detected_phonemes:
                    error_det = {e["detected_phoneme"] for e in wr.errors if e.get("detected_phoneme")}
                    tags = [
                        f'<span class="phoneme-tag {"ph-error" if ph in error_det else "ph-correct"}">'
                        f'{ph}</span>'
                        for ph in wr.detected_phonemes
                    ]
                    st.markdown(" ".join(tags), unsafe_allow_html=True)
                else:
                    st.markdown("*Not detected*")

            # Audio button
            with audio_col:
                st.markdown(f"**Score: {wr.score}/100**")
                audio_bytes = tts_audio_service.word_audio_bytes(wr.word)
                if audio_bytes:
                    st.markdown("🔊 **Correct:**")
                    st.audio(audio_bytes, format="audio/mp3")

            # Error table
            if wr.errors:
                st.markdown("**Errors:**")
                rows = [{
                    "Expected": e.get("expected_phoneme") or "—",
                    "Detected": e.get("detected_phoneme") or "—",
                    "Type":     e.get("error_type", "").capitalize(),
                    "Severity": e.get("severity", "").capitalize(),
                } for e in wr.errors]
                st.table(rows)

            # Suggestion
            if wr.suggestion:
                st.success(f"💡 {wr.suggestion}")

    st.divider()

    # ── Bottom CTA ────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([2, 2, 3])
    with col1:
        if st.button("📊 View Full Report →", type="primary",
                     use_container_width=True, key="report_btn_bottom"):
            st.switch_page("pages/2_Overall_Report.py")
    with col2:
        if st.button("🔁 Try Another Sentence",
                     use_container_width=True, key="retry_btn_bottom"):
            _reset()
            st.rerun()
