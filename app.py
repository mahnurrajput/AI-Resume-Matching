"""
app.py  —  AI Resume–Job Matching System  |  Streamlit UI
==========================================================
Place this file at the project root:

    AI-Resume-Matching/
    ├── app.py            ← this file
    ├── models/
    │   ├── matching_engine.py
    │   ├── skill_analyzer.py
    │   ├── faiss_index.bin
    │   ├── job_metadata.csv
    │   └── ...
    └── data_processed/
        └── jobs_cleaned.csv

Run:
    streamlit run app.py
"""

import os
import sys
import time
import html as html_lib
import textwrap
import streamlit as st
import pdfplumber
from docx import Document

# ── Download model files from HF Hub if not already present ──────────────────
from download_models import download_all

# ── Path setup so matching_engine.py (which lives in models/) can be imported ─
_APP_DIR    = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.join(_APP_DIR, "models")
if _MODELS_DIR not in sys.path:
    sys.path.insert(0, _MODELS_DIR)

# ── Page config  (must be the very first Streamlit call) ──────────────────────
st.set_page_config(
    page_title = "ResumeIQ — AI Job Matching",
    page_icon  = "⚡",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL STYLES
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
/* ── Google Fonts ─────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* ── CSS Variables ────────────────────────────────────────────────────────── */
:root {
    --bg         : #0a0d14;
    --bg2        : #111520;
    --bg3        : #171c2c;
    --border     : #2a314d; /* Slightly lighter border for better card definition */
    --accent     : #5b6ef5;
    --accent2    : #38d9a9;
    --accent3    : #f7c948;
    --danger     : #f4505a;
    --text       : #ffffff; /* Bright white for high contrast */
    --text-muted : #aab3c6; /* Lighter grey so subtext is actually readable */
    --radius     : 14px;
    --radius-sm  : 8px;
}

/* ── Base reset ───────────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family : 'DM Sans', sans-serif;
    background  : var(--bg) !important;
    color       : var(--text) !important;
    overflow-x  : hidden;
}

/* ── Hide Streamlit chrome ───────────────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem !important; }

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background  : var(--bg2) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Typography ──────────────────────────────────────────────────────────── */
h1, h2, h3, h4 { font-family: 'Syne', sans-serif !important; }

/* ── Buttons ─────────────────────────────────────────────────────────────── */
.stButton > button {
    background    : var(--accent) !important;
    color         : #fff !important;
    border        : none !important;
    border-radius : var(--radius-sm) !important;
    font-family   : 'Syne', sans-serif !important;
    font-weight   : 600 !important;
    letter-spacing: 0.03em !important;
    padding       : 0.6rem 1.6rem !important;
    transition    : opacity 0.15s ease !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* ── File uploader ───────────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    border        : 1.5px dashed var(--border) !important;
    border-radius : var(--radius) !important;
    background    : var(--bg2) !important;
    padding       : 1rem !important;
}

/* ── Sliders / selects ───────────────────────────────────────────────────── */
[data-testid="stSlider"] > div > div > div > div { background: var(--accent) !important; }
[data-testid="stSelectbox"] > div > div { background: var(--bg3) !important; border-color: var(--border) !important; }

/* ── Tabs ────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"]  { background: var(--bg2) !important; border-bottom: 1px solid var(--border) !important; gap: 0.25rem; }
.stTabs [data-baseweb="tab"]       { color: var(--text-muted) !important; font-family: 'Syne', sans-serif !important; }
.stTabs [aria-selected="true"]     { color: var(--accent) !important; border-bottom: 2px solid var(--accent) !important; background: transparent !important; }

/* ── Progress bar ────────────────────────────────────────────────────────── */
.stProgress > div > div > div { background: var(--accent) !important; }

/* ── Custom card class used via st.markdown ──────────────────────────────── */
.riq-card {
    background    : var(--bg2);
    border        : 1px solid var(--border);
    border-radius : var(--radius);
    padding       : 1.5rem 1.75rem;
    margin-bottom : 1rem;
    transition    : border-color 0.2s;
    overflow      : hidden;
}
.riq-card:hover { border-color: var(--accent); }

.riq-card-header {
    display       : flex;
    align-items   : flex-start;
    gap           : 1rem;
    margin-bottom : 0.75rem;
}
.riq-card-header > div { min-width: 0; }
.riq-rank {
    font-family   : 'Syne', sans-serif;
    font-size     : 2rem;
    font-weight   : 800;
    color         : rgba(255, 255, 255, 0.23); /* Soft, elegant visibility */
    min-width     : 2.5rem;
    line-height   : 1;
}
.riq-title {
    font-family   : 'Syne', sans-serif;
    font-size     : 1.15rem;
    font-weight   : 700;
    color         : var(--text);
    margin-bottom : 0.15rem;
}
.riq-company { color: var(--text-muted); font-size: 0.9rem; }
.riq-title,
.riq-company,
.hero-sub {
    overflow-wrap : anywhere;
}

.riq-score-bar-wrap {
    background    : var(--bg3);
    border-radius : 99px;
    height        : 6px;
    overflow      : hidden;
    margin        : 0.75rem 0 0.5rem;
}
.riq-score-bar {
    height        : 6px;
    border-radius : 99px;
    background    : linear-gradient(90deg, var(--accent), var(--accent2));
}

.riq-pill {
    display       : inline-flex;
    align-items   : center;
    max-width     : 100%;
    padding       : 0.2rem 0.7rem;
    border-radius : 99px;
    font-size     : 0.75rem;
    font-weight   : 500;
    margin        : 0.15rem;
    word-break    : break-word;
    overflow-wrap : anywhere;
}
.pill-matched  { background: rgba(56,217,169,0.15); color: var(--accent2); border: 1px solid rgba(56,217,169,0.3); }
.pill-missing  { background: rgba(244,80,90,0.12);  color: var(--danger);  border: 1px solid rgba(244,80,90,0.25); }
.pill-extra    { background: rgba(165, 180, 252, 0.15); color: #a5b4fc;  border: 1px solid rgba(165, 180, 252, 0.3); }
.pill-neutral  { background: var(--bg3);            color: var(--text-muted); border: 1px solid var(--border); }

.verdict-strong   { color: var(--accent2); font-weight: 700; }
.verdict-moderate { color: var(--accent3); font-weight: 700; }
.verdict-weak     { color: var(--danger);  font-weight: 700; }

.riq-section-label {
    font-size     : 0.75rem;
    font-weight   : 800;  
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color         : var(--text);
    font-family   : 'Syne', sans-serif;
    margin-bottom : 0.6rem;
}

.riq-stat {
    display: inline-flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 0.1rem;
    margin-right: 1.5rem;
}
.riq-stat-val { font-family: 'Syne', sans-serif; font-size: 1.1rem; font-weight: 700; }
.riq-stat-key { font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.1em; }

.hero-title {
    font-family   : 'Syne', sans-serif;
    font-size     : clamp(2rem, 4vw, 3rem);
    font-weight   : 800;
    line-height   : 1.1;
    background    : linear-gradient(135deg, #fff 30%, var(--accent));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    color         : var(--text-muted);
    font-size     : 1rem;
    margin-top    : 0.5rem;
    max-width     : 500px;
}

.tag-row {
    display       : flex;
    flex-wrap     : wrap;
    gap           : 0.35rem;
    margin-top    : 0.4rem;
    max-width     : 100%;
    min-width     : 0;
}


.tag-row:last-child {
    margin-bottom : 0.5rem; /* Creates breathing room at the bottom of the card */
}

.tag-row .riq-pill {
    flex          : 0 1 auto;
    max-width     : 100%;
}

.riq-summary-bar {
    display       : flex;
    gap           : 2.5rem;
    align-items   : center;
    flex-wrap     : wrap;
    padding       : 1rem 1.5rem;
    background    : var(--bg2);
    border        : 1px solid var(--border);
    border-radius : var(--radius);
    margin        : 1.5rem 0;
}

.riq-summary-bar .riq-stat {
    margin-right  : 0;
}

.result-grid {
    display       : grid;
    grid-template-columns : minmax(0, 1fr) minmax(0, 1fr);
    gap           : 1.25rem;
}

@media (max-width: 900px) {
    .riq-card {
        padding : 1.1rem 1rem;
    }
    .riq-card-header {
        flex-direction : column;
    }
    .riq-card-header > div:last-child {
        align-self : flex-start;
        text-align : left;
        min-width  : 0;
    }
    .riq-summary-bar {
        gap : 1rem 1.5rem;
    }
    .result-grid {
        grid-template-columns : 1fr;
    }
    div[data-testid="stHorizontalBlock"] {
        flex-wrap : wrap;
    }
    div[data-testid="column"] {
        flex       : 1 1 100% !important;
        width      : 100% !important;
        min-width  : 100% !important;
    }
}

.ai-box {
    background    : linear-gradient(135deg, rgba(91,110,245,0.1), rgba(56,217,169,0.06));
    border        : 1px solid rgba(91,110,245,0.35);
    border-radius : var(--radius);
    padding       : 1.25rem 1.5rem;
    margin-top    : 0.75rem;
}
.ai-summary { font-style: italic; color: var(--text); line-height: 1.7; }

.divider { border: none; border-top: 1px solid var(--border); margin: 0.75rem 0; }

/* Spinner color */
.stSpinner > div > div { border-top-color: var(--accent) !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — TEXT EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_pdf(file_obj) -> str:
    text = []
    with pdfplumber.open(file_obj) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text.append(t)
    return "\n".join(text)


def extract_docx(file_obj) -> str:
    doc = Document(file_obj)
    return "\n".join(p.text for p in doc.paragraphs if p.text)


def extract_text(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        return extract_pdf(uploaded_file)
    elif name.endswith(".docx"):
        return extract_docx(uploaded_file)
    else:
        return uploaded_file.read().decode("utf-8", errors="ignore")


# ══════════════════════════════════════════════════════════════════════════════
# CACHED ENGINE LOADER
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_engine():
    """Load the MatchingEngine once per server process — heavy (~7–8 sec)."""
    from matching_engine import MatchingEngine
    return MatchingEngine()


# ══════════════════════════════════════════════════════════════════════════════
# RENDER HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def score_color(score: float) -> str:
    if score >= 0.70:
        return "#38d9a9"
    if score >= 0.55:
        return "#f7c948"
    return "#f4505a"


def verdict_html(v: str) -> str:
    labels = {
        "strong_fit"  : ('<span class="verdict-strong">Strong Fit</span>',   "🟢"),
        "moderate_fit": ('<span class="verdict-moderate">Moderate Fit</span>', "🟡"),
        "weak_fit"    : ('<span class="verdict-weak">Weak Fit</span>',        "🔴"),
    }
    return labels.get(v, ("", ""))[0]


def severity_pill(sev: str) -> str:
    if sev == "low":
        return '<span class="riq-pill pill-matched">Low Gap</span>'
    if sev == "medium":
        return '<span class="riq-pill pill-neutral">Medium Gap</span>'
    if sev == "high":
        return '<span class="riq-pill pill-missing">High Gap</span>'
    return '<span class="riq-pill pill-neutral">–</span>'


def pills(items, css_class) -> str:
    return "".join(
        f'<span class="riq-pill {css_class}">{html_lib.escape(str(i))}</span>'
        for i in items
        if i not in (None, "")
    )


def esc(value) -> str:
    return html_lib.escape("" if value is None else str(value))


def render_result_card(r, expanded_idx: int, card_idx: int):
    sg = r.skill_gap
    score_pct = int(r.score * 100)
    bar_color = score_color(r.score)

    remote_badge = '<span class="riq-pill pill-matched">Remote OK</span>' if r.remote_allowed == 1.0 else ""
    salary_str = f'${r.salary_min:,.0f} – ${r.salary_max:,.0f}' if (r.salary_min and r.salary_max) else ""
    
    meta_pills = ""
    if r.experience_level and str(r.experience_level) != "nan":
        meta_pills += f'<span class="riq-pill pill-neutral">{esc(r.experience_level)}</span>'
    if r.work_type and str(r.work_type) != "nan":
        meta_pills += f'<span class="riq-pill pill-neutral">{esc(r.work_type)}</span>'
    meta_pills += remote_badge

    overlap_html = severity_pill(sg.gap_severity) if sg else ""
    verdict_line = verdict_html(sg.candidacy_verdict) if (sg and sg.ai_available) else ""
    
    matched_pills = pills(sg.flat_matched()[:8], "pill-matched") if sg else ""
    missing_pills = pills(sg.flat_missing()[:8], "pill-missing") if sg else ""

    # CRITICAL: Do NOT indent these lines. They must start at the very beginning of the line.
    card_html = f"""
<div class="riq-card">
<div class="riq-card-header">
<div class="riq-rank">#{esc(r.rank)}</div>
<div style="flex:1; min-width:0;">
<div class="riq-title">{esc(r.title) or '–'}</div>
<div class="riq-company">{esc(r.company) or 'Unknown'} &nbsp;·&nbsp; {esc(r.location) or '–'}</div>
{f'<div style="color:#7a82a0;font-size:0.82rem;margin-top:0.2rem">{salary_str}</div>' if salary_str else ''}
</div>
<div style="text-align:right; min-width:80px;">
<div style="font-family:sans-serif;font-size:1.6rem;font-weight:800;color:{bar_color}">{score_pct}<span style="font-size:0.9rem;font-weight:400">%</span></div>
<div style="font-size:0.68rem;color:#7a82a0;letter-spacing:0.08em;text-transform:uppercase">match</div>
</div>
</div>
<div class="riq-score-bar-wrap">
<div class="riq-score-bar" style="width:{score_pct}%; background-color:{bar_color};"></div>
</div>
{f'<div style="display:flex;justify-content:flex-end;margin-top:0.35rem">{verdict_line}</div>' if verdict_line else ''}
<div class="tag-row" style="margin-top:0.5rem">
{meta_pills}
{overlap_html}
</div>
<div class="tag-row" style="margin-top:0.6rem">{matched_pills}</div>
<div class="tag-row">{missing_pills}</div>
</div>
"""
    st.markdown(card_html, unsafe_allow_html=True)

    if sg and (sg.ai_available or sg.flat_matched() or sg.flat_missing()):
        with st.expander(f"View full analysis — {r.title}", expanded=(card_idx == expanded_idx)):
            _render_detail_panel(r, sg)


def _render_detail_panel(r, sg):
    """Detailed skill gap breakdown inside the expander."""

    # ── AI summary box ────────────────────────────────────────────────────────
    if sg.ai_available and sg.executive_summary:
        # ZERO indentation to prevent Streamlit code-block bugs
        ai_box_html = f"""
<div class="ai-box">
<div class="riq-title">AI Hiring Manager Assessment</div>
<div class="ai-summary" style="color:var(--text); font-size:1.05rem; line-height:1.6; font-style:italic;">"{esc(sg.executive_summary)}"</div>
<div style="margin-top:1.5rem; display:flex; gap:1.5rem;">
<span class="riq-stat">
<span class="riq-stat-val">{esc(sg.candidacy_verdict.replace("_"," ").title())}</span>
<span class="riq-stat-key">Verdict</span>
</span>
<span class="riq-stat">
<span class="riq-stat-val">{esc(sg.verdict_confidence.title())}</span>
<span class="riq-stat-key">Confidence</span>
</span>
<span class="riq-stat">
<span class="riq-stat-val">{int(sg.overlap_score * 100)}%</span>
<span class="riq-stat-key">Skill Overlap</span>
</span>
</div>
<div style="margin-top:1.5rem;">
<span class="riq-stat">
<span class="riq-stat-val">{esc(sg.time_to_ready or '–')}</span>
<span class="riq-stat-key">Time to Ready</span>
</span>
</div>
</div>
"""
        st.markdown(ai_box_html, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Matched skills
        matched = sg.structured.matched
        if matched:
            st.markdown('<div class="riq-title" style="margin-top:1rem">Matching Skills</div>', unsafe_allow_html=True)
            for cat, skills in matched.items():
                # Brightened blue (#8291ff) and added font-weight:700
                st.markdown(f'<div style="font-size:0.75rem; font-weight:700; color:#8291ff; margin:0.8rem 0 0.3rem; text-transform:uppercase; letter-spacing:0.1em">{esc(cat.replace("_"," "))}</div>', unsafe_allow_html=True)
                st.markdown('<div class="tag-row">' + pills(skills, "pill-matched") + '</div>', unsafe_allow_html=True)

        # Extra / transferable skills
        extra = sg.structured.extra
        if extra:
            st.markdown('<div class="riq-title" style="margin-top:1.5rem">Extra Skills</div>', unsafe_allow_html=True)
            all_extra = sg.flat_extra()
            st.markdown('<div class="tag-row">' + pills(all_extra[:20], "pill-extra") + '</div>', unsafe_allow_html=True)

    with col2:
        # Missing skills
        missing = sg.structured.missing
        if missing:
            st.markdown('<div class="riq-title" style="margin-top:1rem">Missing Skills</div>', unsafe_allow_html=True)
            for cat, skills in missing.items():
                # Brightened red (#ff6b75) and added font-weight:700
                st.markdown(f'<div style="font-size:0.75rem; font-weight:700; color:#ff6b75; margin:0.8rem 0 0.3rem; text-transform:uppercase; letter-spacing:0.1em">{esc(cat.replace("_"," "))}</div>', unsafe_allow_html=True)
                st.markdown('<div class="tag-row">' + pills(skills, "pill-missing") + '</div>', unsafe_allow_html=True)

        # Dealbreakers
        if sg.ai_available and sg.dealbreaker_skills:
            st.markdown('<div class="riq-title" style="margin-top:1.5rem">Dealbreakers</div>', unsafe_allow_html=True)
            # Wrapped in tag-row to ensure flex wrapping
            dealbreaker_pills = "".join([f'<span class="riq-pill pill-missing">{esc(d)}</span>' for d in sg.dealbreaker_skills])
            st.markdown(f'<div class="tag-row">{dealbreaker_pills}</div>', unsafe_allow_html=True)

    # ── Strengths and risks ───────────────────────────────────────────────────
    if sg.ai_available and (sg.strengths or sg.hiring_risks):
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            if sg.strengths:
                st.markdown('<div class="riq-title">Strengths for this role</div>', unsafe_allow_html=True)
                for s in sg.strengths:
                    # Changed text to pure white (#ffffff)
                    st.markdown(f'<div style="font-size:0.9rem; color:#ffffff; margin:0.5rem 0; padding-left:0.8rem; border-left:2px solid #38d9a9">{esc(s)}</div>', unsafe_allow_html=True)
        with c2:
            if sg.hiring_risks:
                st.markdown('<div class="riq-title">Hiring risks</div>', unsafe_allow_html=True)
                for risk in sg.hiring_risks:
                    # Changed text to pure white (#ffffff)
                    st.markdown(f'<div style="font-size:0.9rem; color:#ffffff; margin:0.5rem 0; padding-left:0.8rem; border-left:2px solid #f4505a">{esc(risk)}</div>', unsafe_allow_html=True)

    # ── Learning path ─────────────────────────────────────────────────────────
    if sg.ai_available and sg.learning_path:
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown('<div class="riq-title">Learning Path to Close the Gap</div>', unsafe_allow_html=True)
        for i, step in enumerate(sg.learning_path, 1):
            # ZERO indentation. Brightened numbers to #8291ff, text to #ffffff
            path_html = f"""
<div style="display:flex; gap:0.8rem; align-items:flex-start; margin:0.6rem 0; padding:0.5rem; background:var(--bg3); border-radius:8px;">
<div style="font-family:'Syne',sans-serif; font-weight:800; font-size:1.1rem; color:#8291ff; min-width:1.5rem">{i}</div>
<div style="font-size:0.92rem; color:#ffffff; line-height:1.6">{esc(step)}</div>
</div>
"""
            st.markdown(path_html, unsafe_allow_html=True)

    # ── Compensatable gaps ───────────────────────────────────────────────────
    if sg.ai_available and sg.compensatable_gaps:
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown('<div class="riq-title">Gaps You Can Compensate For</div>', unsafe_allow_html=True)
        for g in sg.compensatable_gaps:
            # Changed text to pure white (#ffffff)
            st.markdown(f'<div style="font-size:0.9rem; color:#ffffff; margin:0.4rem 0; padding:0.6rem 0.8rem; background:var(--bg3); border-radius:6px">{esc(g)}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.markdown("""
<div style="padding:0.5rem 0 1.5rem">
  <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;background:linear-gradient(135deg,#fff,#5b6ef5);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text">ResumeIQ</div>
  <div style="font-size:0.75rem;color:#7a82a0;letter-spacing:0.1em;text-transform:uppercase;margin-top:0.15rem">AI Job Matching</div>
</div>
""", unsafe_allow_html=True)

        st.markdown("#### ⚙️ Match Settings")

        top_k = st.slider(
            "Number of results",
            min_value=3, max_value=20, value=5, step=1,
            help="How many job matches to retrieve."
        )

        enable_skill_gap = st.toggle(
            "Enable Skill Gap Analysis",
            value=True,
            help="Runs Stage 1 structured extraction (spaCy + taxonomy). Adds ~3–5 sec per result."
        )

        enable_ai = False
        if enable_skill_gap:
            enable_ai = st.toggle(
                "Enable AI Reasoning (Gemini)",
                value=True,
                help="Activates Stage 2 Gemini reasoning for verdicts, learning paths, and hiring risks. Requires GEMINI_API_KEY."
            )

        st.markdown("---")
        st.markdown("#### 📊 System Stats")
        st.markdown("""
<div style="font-size:0.82rem;color:#7a82a0;line-height:2">
  108,702 job postings indexed<br>
  11,654 training resumes<br>
  SBERT all-MiniLM-L6-v2<br>
  FAISS IndexFlatIP (exact)<br>
  400+ skill taxonomy
</div>
""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
<div style="font-size:0.75rem;color:#7a82a0">
  Set <code>GEMINI_API_KEY</code> in <code>.env</code><br>
  to enable Stage 2 AI reasoning.
</div>
""", unsafe_allow_html=True)

    return top_k, enable_skill_gap, enable_ai


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

def main():

    download_all()
    
    top_k, enable_skill_gap, enable_ai = render_sidebar()

    # ── Hero header ───────────────────────────────────────────────────────────
    st.markdown("""
<div style="margin-bottom:2rem">
  <div class="hero-title">Find your perfect job match.</div>
  <div class="hero-sub">Upload your resume. We'll compare it against 108,702 real LinkedIn job postings using AI-powered semantic matching and skill gap analysis.</div>
</div>
""", unsafe_allow_html=True)

    # ── Upload section ────────────────────────────────────────────────────────
    tab_upload, tab_paste = st.tabs(["📄 Upload Resume", "✏️  Paste Text"])

    resume_text = ""

    with tab_upload:
        uploaded = st.file_uploader(
            "Drop your resume here",
            type=["pdf", "docx", "txt"],
            label_visibility="collapsed",
        )
        if uploaded:
            with st.spinner("Extracting text…"):
                try:
                    resume_text = extract_text(uploaded)
                    wc = len(resume_text.split())
                    st.success(f"✅ Extracted **{wc:,}** words from **{uploaded.name}**")
                    with st.expander("Preview extracted text"):
                        st.text(textwrap.fill(resume_text[:2000], width=80) + ("…" if len(resume_text) > 2000 else ""))
                except Exception as e:
                    st.error(f"Could not parse file: {e}")

    with tab_paste:
        resume_text_paste = st.text_area(
            "Paste resume text here",
            height=280,
            placeholder="Paste your resume text directly here…",
            label_visibility="collapsed",
        )
        if resume_text_paste.strip():
            resume_text = resume_text_paste

    # ── Match button ──────────────────────────────────────────────────────────
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    run_col, _ = st.columns([1, 4])
    with run_col:
        run_match = st.button("⚡ Find Matches", use_container_width=True)

    # ── Session state ─────────────────────────────────────────────────────────
    if "results" not in st.session_state:
        st.session_state.results      = None
        st.session_state.elapsed      = 0
        st.session_state.resume_text  = ""

    # ── Run matching ──────────────────────────────────────────────────────────
    if run_match:
        if not resume_text or not resume_text.strip():
            st.warning("Please upload a resume or paste your text before running.")
        elif len(resume_text.split()) < 30:
            st.warning("Resume seems too short (< 30 words). Please provide more content.")
        else:
            st.session_state.resume_text = resume_text

            # Load engine (cached — only slow on first ever call)
            with st.spinner("Loading matching engine… (first load may take ~10 seconds)"):
                try:
                    engine = load_engine()
                except Exception as e:
                    st.error(f"Failed to load engine: {e}")
                    st.stop()

            # Run matching
            progress_bar = st.progress(0, text="Embedding resume…")
            t0 = time.time()

            try:
                progress_bar.progress(25, text="Searching 108,702 jobs…")
                results = engine.match(
                    resume_text,
                    top_k            = top_k,
                    enable_skill_gap = enable_skill_gap,
                    enable_ai        = enable_ai,
                )
                elapsed = time.time() - t0
                progress_bar.progress(100, text="Done!")
                time.sleep(0.4)
                progress_bar.empty()

                st.session_state.results = results
                st.session_state.elapsed = elapsed

            except Exception as e:
                progress_bar.empty()
                st.error(f"Matching failed: {e}")
                import traceback; st.code(traceback.format_exc())

    # ── Display results ───────────────────────────────────────────────────────
    results = st.session_state.results
    if results:
        elapsed = st.session_state.elapsed

        # ── Summary bar ───────────────────────────────────────────────────────
        avg_score = sum(r.score for r in results) / len(results)
        has_ai    = any(r.skill_gap and r.skill_gap.ai_available for r in results)

        st.markdown(f"""
    <div class="riq-summary-bar">
  <span class="riq-stat">
    <span class="riq-stat-val">{len(results)}</span>
    <span class="riq-stat-key">Matches Found</span>
  </span>
  <span class="riq-stat">
    <span class="riq-stat-val">{int(avg_score * 100)}%</span>
    <span class="riq-stat-key">Avg. Score</span>
  </span>
  <span class="riq-stat">
    <span class="riq-stat-val">{elapsed:.1f}s</span>
    <span class="riq-stat-key">Run Time</span>
  </span>
  <span class="riq-stat">
    <span class="riq-stat-val">{'AI + Structured' if has_ai else 'Structured' if enable_skill_gap else 'Semantic Only'}</span>
    <span class="riq-stat-key">Analysis Mode</span>
  </span>
</div>
""", unsafe_allow_html=True)

        # ── Sort / filter controls ─────────────────────────────────────────────
        fcol1, fcol2, fcol3 = st.columns([2, 2, 2])
        with fcol1:
            sort_by = st.selectbox(
                "Sort by",
                ["Match Score", "Skill Overlap (if available)"],
                label_visibility="collapsed",
            )
        with fcol2:
            filter_remote = st.selectbox(
                "Remote filter",
                ["All Jobs", "Remote Only"],
                label_visibility="collapsed",
            )
        with fcol3:
            filter_level = st.selectbox(
                "Experience level",
                ["All Levels"] + sorted({r.experience_level for r in results if r.experience_level not in ("nan", "", None)}),
                label_visibility="collapsed",
            )

        # Apply filters
        display = list(results)

        if filter_remote == "Remote Only":
            display = [r for r in display if r.remote_allowed == 1.0]

        if filter_level != "All Levels":
            display = [r for r in display if r.experience_level == filter_level]

        if sort_by == "Skill Overlap (if available)":
            display.sort(
                key=lambda r: r.skill_gap.overlap_score if r.skill_gap else 0,
                reverse=True,
            )

        if not display:
            st.info("No results match the current filters.")
        else:
            st.markdown(f"<div style='color:#7a82a0;font-size:0.82rem;margin-bottom:0.5rem'>Showing {len(display)} result{'s' if len(display)!=1 else ''}</div>", unsafe_allow_html=True)

            for i, r in enumerate(display):
                render_result_card(r, expanded_idx=-1, card_idx=i)

    elif not run_match:
        # ── Empty state ───────────────────────────────────────────────────────
        st.markdown("""
<div style="text-align:center;padding:4rem 2rem;color:#7a82a0">
  <div style="font-size:3rem;margin-bottom:1rem">📋</div>
  <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:600;color:#4a5270">Upload your resume to get started</div>
  <div style="font-size:0.88rem;margin-top:0.5rem">Supports PDF, DOCX, and plain text files</div>
  <div style="margin-top:2rem;display:flex;justify-content:center;gap:2rem;flex-wrap:wrap">
    <div style="padding:1rem 1.5rem;background:var(--bg2);border:1px solid var(--border);border-radius:var(--radius);text-align:left;min-width:160px">
      <div style="font-size:1.5rem">🔍</div>
      <div style="font-family:'Syne',sans-serif;font-weight:600;margin:0.4rem 0 0.2rem">Semantic Search</div>
      <div style="font-size:0.8rem;color:#7a82a0">SBERT matches meaning,<br>not just keywords</div>
    </div>
    <div style="padding:1rem 1.5rem;background:var(--bg2);border:1px solid var(--border);border-radius:var(--radius);text-align:left;min-width:160px">
      <div style="font-size:1.5rem">🧠</div>
      <div style="font-family:'Syne',sans-serif;font-weight:600;margin:0.4rem 0 0.2rem">Skill Analysis</div>
      <div style="font-size:0.8rem;color:#7a82a0">400+ skill taxonomy<br>+ spaCy NER</div>
    </div>
    <div style="padding:1rem 1.5rem;background:var(--bg2);border:1px solid var(--border);border-radius:var(--radius);text-align:left;min-width:160px">
      <div style="font-size:1.5rem">✨</div>
      <div style="font-family:'Syne',sans-serif;font-weight:600;margin:0.4rem 0 0.2rem">AI Reasoning</div>
      <div style="font-size:0.8rem;color:#7a82a0">Gemini gives hiring<br>manager insights</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
