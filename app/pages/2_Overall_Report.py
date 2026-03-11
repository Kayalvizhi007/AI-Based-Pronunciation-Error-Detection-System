"""
Page 2 – Overall Pronunciation Report
---------------------------------------
Shown after the word-by-word analysis on page 1.
Provides a rich, emoji-filled overall score, strengths/weaknesses,
detailed improvement plan, and balloons for scores ≥ 75.

Run via Streamlit multipage:
    streamlit run app/streamlit_app.py
Then navigate to this page from the main page's "View Full Report" button.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import streamlit as st

st.set_page_config(
    page_title="Overall Pronunciation Report",
    page_icon="📊",
    layout="centered",
)

# ── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.big-score {
    font-size: 6rem;
    font-weight: 900;
    line-height: 1;
    text-align: center;
}
.score-ring {
    border-radius: 50%;
    width: 180px;
    height: 180px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 12px auto;
    border: 8px solid;
}
.grade-label {
    font-size: 1.5rem;
    font-weight: 700;
    text-align: center;
    margin-top: 4px;
}
.phoneme-tag {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    margin: 3px;
    font-family: monospace;
    font-size: 0.9rem;
    font-weight: 700;
}
.ph-good    { background:#d5f5e3; color:#1a7a45; }
.ph-bad     { background:#fde8e8; color:#c0392b; }
.ph-partial { background:#fef9e7; color:#b7770d; }
.stat-box {
    background: #f8f9fa;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}
.stat-number { font-size: 2rem; font-weight: 800; }
.stat-label  { font-size: 0.8rem; color: #888; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)

# ── Guard: need a report in session state ──────────────────────────────────
if "report" not in st.session_state or st.session_state.report is None:
    st.warning("No report found. Please record and analyse a sentence first.")
    if st.button("← Go to Pronunciation Checker"):
        st.switch_page("streamlit_app.py")
    st.stop()

report = st.session_state.report
score  = report.overall_score


# ── Score band config ──────────────────────────────────────────────────────

def _band(s: int) -> dict:
    if s >= 90:
        return dict(
            color="#1a9e5a", emoji="🏆",
            title="Outstanding!",
            subtitle="Your pronunciation is near-native. Excellent work!",
            cheers="🎉🌟🎊✨🥇🎯🔥"
        )
    if s >= 75:
        return dict(
            color="#27ae60", emoji="🌟",
            title="Great Job!",
            subtitle="You're pronouncing English very well. Keep it up!",
            cheers="🎉👏🌟😄🎊✨"
        )
    if s >= 60:
        return dict(
            color="#f39c12", emoji="👍",
            title="Good Progress",
            subtitle="Solid effort — a few sounds need polish but you're on the right track.",
            cheers=""
        )
    if s >= 40:
        return dict(
            color="#e67e22", emoji="💪",
            title="Keep Practising",
            subtitle="You're building the foundation. Focus on the sounds below.",
            cheers=""
        )
    return dict(
        color="#e74c3c", emoji="🔄",
        title="Let's Work on This",
        subtitle="Don't worry — even small improvements show real progress!",
        cheers=""
    )


band = _band(score)

# ── Balloons for high scores ───────────────────────────────────────────────
if score >= 75:
    st.balloons()

# ── Header ─────────────────────────────────────────────────────────────────
st.title("📊 Overall Pronunciation Report")
st.markdown(f'**Sentence you practised:** *"{report.sentence}"*')
st.divider()

# ── Score display ──────────────────────────────────────────────────────────
col_score, col_info = st.columns([1, 2])

with col_score:
    st.markdown(
        f'<div class="score-ring" style="border-color:{band["color"]}">'
        f'<div class="big-score" style="color:{band["color"]}">{score}</div>'
        f'</div>'
        f'<div class="grade-label" style="color:{band["color"]}">'
        f'{band["emoji"]} {band["title"]}</div>',
        unsafe_allow_html=True,
    )

with col_info:
    st.markdown(f"### {band['subtitle']}")
    if band["cheers"]:
        st.markdown(f"<div style='font-size:2rem;letter-spacing:6px'>{band['cheers']}</div>",
                    unsafe_allow_html=True)
    st.markdown(f"**Score: {score} / 100**")

    # Quick stats
    wrs       = report.word_reports
    n_perfect = sum(1 for w in wrs if w.score >= 95)
    n_good    = sum(1 for w in wrs if 70 <= w.score < 95)
    n_needs   = sum(1 for w in wrs if w.score < 70)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f'<div class="stat-box">'
            f'<div class="stat-number" style="color:#27ae60">{n_perfect}</div>'
            f'<div class="stat-label">Perfect words</div></div>',
            unsafe_allow_html=True)
    with c2:
        st.markdown(
            f'<div class="stat-box">'
            f'<div class="stat-number" style="color:#f39c12">{n_good}</div>'
            f'<div class="stat-label">Near-correct</div></div>',
            unsafe_allow_html=True)
    with c3:
        st.markdown(
            f'<div class="stat-box">'
            f'<div class="stat-number" style="color:#e74c3c">{n_needs}</div>'
            f'<div class="stat-label">Needs work</div></div>',
            unsafe_allow_html=True)

st.divider()

# ── Coach's narrative summary ──────────────────────────────────────────────
if report.overall_suggestion:
    st.subheader("🎓 Coach's Feedback")
    st.info(report.overall_suggestion)
    st.divider()

# ── Word score bar chart ───────────────────────────────────────────────────
st.subheader("📈 Word-by-Word Score Breakdown")

import pandas as pd

word_data = pd.DataFrame({
    "Word":  [wr.word for wr in wrs],
    "Score": [wr.score for wr in wrs],
})

# Color map: ≥80 green, 50-79 orange, <50 red
def _bar_color(s):
    if s >= 80: return "#27ae60"
    if s >= 50: return "#f39c12"
    return "#e74c3c"

# Render as horizontal bars using st.progress per word
for wr in wrs:
    icon = "✅" if wr.score >= 80 else ("⚠️" if wr.score >= 50 else "❌")
    col_word, col_bar, col_num = st.columns([2, 6, 1])
    with col_word:
        st.markdown(f"**{icon} {wr.word}**")
    with col_bar:
        st.progress(wr.score / 100)
    with col_num:
        st.markdown(f"**{wr.score}**")

st.divider()

# ── Strengths & Weaknesses ─────────────────────────────────────────────────
st.subheader("💪 Strengths & Areas to Improve")

strong_words = [wr for wr in wrs if wr.score >= 80]
weak_words   = [wr for wr in wrs if wr.score < 70]

col_s, col_w = st.columns(2)

with col_s:
    st.markdown("#### ✅ You nailed these:")
    if strong_words:
        for wr in strong_words:
            st.success(f"**{wr.word}** — {wr.score}/100")
    else:
        st.info("Keep practising to build confident words!")

with col_w:
    st.markdown("#### 🔧 Focus on these:")
    if weak_words:
        for wr in weak_words:
            st.error(f"**{wr.word}** — {wr.score}/100")
    else:
        st.success("No major problem words — great!")

st.divider()

# ── Phoneme error deep-dive ────────────────────────────────────────────────
all_errors = [(wr.word, e) for wr in wrs for e in wr.errors]

if all_errors:
    st.subheader("🔬 Phoneme Error Deep-Dive")

    # Group by error type
    from collections import Counter, defaultdict

    by_type: dict = defaultdict(list)
    for word, e in all_errors:
        by_type[e.get("error_type", "unknown")].append((word, e))

    if "substitution" in by_type:
        with st.expander(f"🔁 Substitutions ({len(by_type['substitution'])} errors)", expanded=True):
            st.markdown("These are sounds you replaced with a different sound:")
            for word, e in by_type["substitution"]:
                exp = e.get("expected_phoneme", "?")
                det = e.get("detected_phoneme", "?")
                sev = e.get("severity", "").capitalize()
                sev_color = {"Minor": "#f39c12", "Moderate": "#e67e22", "Severe": "#e74c3c"}.get(sev, "#888")
                st.markdown(
                    f'**{word}**: '
                    f'<span class="phoneme-tag ph-good">/{exp}/</span> expected → '
                    f'<span class="phoneme-tag ph-bad">/{det}/</span> detected &nbsp;'
                    f'<span style="color:{sev_color};font-size:0.8rem">({sev})</span>',
                    unsafe_allow_html=True,
                )

    if "deletion" in by_type:
        with st.expander(f"🚫 Deletions ({len(by_type['deletion'])} errors)"):
            st.markdown("These are sounds you skipped entirely:")
            for word, e in by_type["deletion"]:
                exp = e.get("expected_phoneme", "?")
                st.markdown(
                    f'**{word}**: missing '
                    f'<span class="phoneme-tag ph-bad">/{exp}/</span>',
                    unsafe_allow_html=True,
                )

    if "insertion" in by_type:
        with st.expander(f"➕ Insertions ({len(by_type['insertion'])} errors)"):
            st.markdown("These are extra sounds you added:")
            for word, e in by_type["insertion"]:
                det = e.get("detected_phoneme", "?")
                st.markdown(
                    f'**{word}**: extra '
                    f'<span class="phoneme-tag ph-partial">/{det}/</span>',
                    unsafe_allow_html=True,
                )

    st.divider()

# ── Personalised improvement plan ─────────────────────────────────────────
if weak_words:
    st.subheader("🗺️ Your Personalised Improvement Plan")

    for wr in weak_words:
        with st.expander(f"📌 How to improve: **{wr.word.upper()}** ({wr.score}/100)"):
            # Phoneme comparison
            ph_col, det_col = st.columns(2)
            with ph_col:
                st.markdown("**Target phonemes:**")
                error_exp = {e["expected_phoneme"] for e in wr.errors if e.get("expected_phoneme")}
                tags = []
                for ph in wr.expected_phonemes:
                    css = "ph-bad" if ph in error_exp else "ph-good"
                    tags.append(f'<span class="phoneme-tag {css}">{ph}</span>')
                st.markdown(" ".join(tags), unsafe_allow_html=True)
            with det_col:
                st.markdown("**What you produced:**")
                if wr.detected_phonemes:
                    error_det = {e["detected_phoneme"] for e in wr.errors if e.get("detected_phoneme")}
                    tags = []
                    for ph in wr.detected_phonemes:
                        css = "ph-bad" if ph in error_det else "ph-good"
                        tags.append(f'<span class="phoneme-tag {css}">{ph}</span>')
                    st.markdown(" ".join(tags), unsafe_allow_html=True)
                else:
                    st.markdown("*Not detected*")

            if wr.suggestion:
                st.markdown("**💡 Tutor tip:**")
                st.success(wr.suggestion)

            # Correct pronunciation audio
            try:
                from services import tts_audio_service
                audio = tts_audio_service.word_audio_bytes(wr.word)
                if audio:
                    st.markdown("**🔊 Hear the correct pronunciation:**")
                    st.audio(audio, format="audio/mp3")
            except Exception:
                pass

    st.divider()

# ── Common error patterns ─────────────────────────────────────────────────
if all_errors:
    st.subheader("📊 Most Common Phoneme Errors")
    from collections import Counter
    counts = Counter(
        e.get("expected_phoneme", "?")
        for _, e in all_errors
        if e.get("expected_phoneme")
    )
    if counts:
        st.markdown("These phonemes caused the most trouble in this session:")
        for ph, cnt in counts.most_common(5):
            col_ph, col_bar, col_cnt = st.columns([1, 7, 1])
            with col_ph:
                st.markdown(f'<span class="phoneme-tag ph-bad">/{ph}/</span>',
                            unsafe_allow_html=True)
            with col_bar:
                st.progress(min(cnt / max(counts.values()), 1.0))
            with col_cnt:
                st.markdown(f"**{cnt}×**")

    st.divider()

# ── Final motivation message ───────────────────────────────────────────────
if score >= 90:
    st.success("🏆 **Outstanding performance!** You're pronouncing English at a near-native level. "
               "Try longer, more complex sentences to keep challenging yourself!")
elif score >= 75:
    st.success(f"🌟 **Well done!** A score of {score}/100 shows real fluency. "
               "Focus on the highlighted sounds and you'll reach near-native very soon!")
elif score >= 60:
    st.info(f"👍 **Good effort!** {score}/100 is a solid base. "
            "Practise the words in your improvement plan daily — even 5 minutes makes a difference.")
elif score >= 40:
    st.warning(f"💪 **Keep going!** {score}/100 — you're in the learning zone. "
               "Slow down, exaggerate each sound, and record yourself comparing to the audio.")
else:
    st.warning(f"🔄 **{score}/100 — Every expert started here.** "
               "Start with 2–3 problem words from the plan above and practise them in isolation.")

st.markdown("")

# ── Navigation ────────────────────────────────────────────────────────────
col_back, col_new, _ = st.columns([2, 2, 3])
with col_back:
    if st.button("← Back to Analysis", use_container_width=True):
        st.switch_page("streamlit_app.py")
with col_new:
    if st.button("🔁 Try Another Sentence", type="primary", use_container_width=True):
        # Reset and go back to main page
        if "stage" in st.session_state:
            st.session_state.stage   = "record"
            st.session_state.report  = None
            st.session_state.corrected = ""
            st.session_state.raw_transcript = ""
            from app.analyzer import PronunciationAnalyzer
            st.session_state.analyzer = PronunciationAnalyzer()
        st.switch_page("streamlit_app.py")
