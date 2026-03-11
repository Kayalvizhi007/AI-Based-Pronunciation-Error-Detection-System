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
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import streamlit as st

try:
    from infrastructure.logging_config import configure_logging
    configure_logging()
except Exception:
    logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

from app.analyzer import PronunciationAnalyzer, PronunciationReport, WordReport
from services import tts_audio_service

st.set_page_config(
    page_title="Pronunciation Checker",
    page_icon="🗣️",
    layout="wide",
)

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
    st.session_state.stage            = "record"


if "stage" not in st.session_state:
    _reset()


def _save_audio(uploaded) -> Path:
    data = uploaded.read()
    tmp  = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.write(data)
    tmp.close()
    return Path(tmp.name)


analyzer: PronunciationAnalyzer = st.session_state.analyzer

# ── Header ─────────────────────────────────────────────────────────────────
st.title("🗣️ Pronunciation Checker")
st.caption("Record → get instant phoneme-level feedback on every word.")
st.divider()

# ==========================================================================
# STAGE 1 – RECORD
# ==========================================================================
if st.session_state.stage == "record":
    st.subheader("Step 1 — Record your sentence")
    st.info("Click the mic, speak any English sentence clearly, then stop.")

    audio_input = st.audio_input("🎙️ Click to record", key="audio_recorder")

    if audio_input is not None:
        with st.spinner("Transcribing…"):
            try:
                audio_path = _save_audio(audio_input)
                st.session_state.audio_path    = audio_path
                raw = analyzer.transcribe(audio_path)
                st.session_state.raw_transcript = raw
                st.session_state.corrected      = analyzer.correct_transcript(raw)
                st.session_state.stage          = "confirm"
                st.rerun()
            except Exception as exc:
                logger.exception("Transcription failed")
                st.error(f"Transcription failed: {exc}")

# ==========================================================================
# STAGE 2 – CONFIRM
# ==========================================================================
elif st.session_state.stage == "confirm":
    st.subheader("Step 2 — Confirm what you said")

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
                    report = analyzer.analyze(
                        audio_path=st.session_state.audio_path,
                        sentence=st.session_state.corrected,
                    )
                    st.session_state.report = report
                    st.session_state.stage  = "report"
                    st.rerun()
                except Exception as exc:
                    logger.exception("Analysis failed")
                    st.error(f"Analysis failed: {exc}")
    with col_r:
        if st.button("🔄 Re-record", use_container_width=True):
            _reset()
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
