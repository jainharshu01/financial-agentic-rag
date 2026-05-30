"""
app.py

Financial Agentic RAG — Main Streamlit UI.

Design philosophy: the query is the only input. Company and year are
auto-detected from query text. No sidebar dropdowns to fight with.
"""

import time
import re
import streamlit as st

from src.retrieval.baseline_rag import baseline_answer
from src.agent.agentic_rag import agentic_answer
from src.agent.query_classifier import classify_query
from src.agent.router import build_retrieval_strategy
from src.agent.company_parser import extract_companies
from src.agent.query_parser import extract_years
from src.utils.sec_urls import get_filing_url


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Financial Agentic RAG",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=JetBrains+Mono:wght@400;500&display=swap');

.stApp { background-color: #faf8f5 !important; font-family: 'DM Sans', sans-serif !important; }
header[data-testid="stHeader"] { background: transparent !important; }
.block-container { padding-top: 2rem !important; max-width: 1200px !important; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f0ede8 0%, #e8e4dd 100%) !important;
    border-right: 1px solid #ddd8d0 !important;
}

h1 { font-family: 'DM Sans', sans-serif !important; font-weight: 700 !important; color: #2d2a26 !important; letter-spacing: -0.03em !important; font-size: 2.2rem !important; }
h2, h3, h4 { font-family: 'DM Sans', sans-serif !important; font-weight: 600 !important; color: #2d2a26 !important; }
p, li, span, div { font-family: 'DM Sans', sans-serif !important; }

/* PRIMARY (Analyze button) */
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #7ba7cc 0%, #9b8ec4 100%) !important;
    color: white !important; font-weight: 600 !important; border: none !important;
    border-radius: 10px !important; padding: 0.55rem 1.8rem !important;
    box-shadow: 0 2px 10px rgba(123, 167, 204, 0.25) !important;
    transition: all 0.25s ease !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 4px 18px rgba(123, 167, 204, 0.35) !important;
    transform: translateY(-1px) !important;
}

/* SECONDARY (example chips) — wraps text, fits any length */
.stButton > button[kind="secondary"], .stButton > button[data-testid="stBaseButton-secondary"] {
    background: #ffffff !important;
    color: #5a554e !important;
    border: 1px solid #e0dbd5 !important;
    border-radius: 14px !important;
    padding: 0.55rem 1rem !important;
    font-size: 0.82rem !important;
    font-weight: 400 !important;
    box-shadow: none !important;
    transition: all 0.2s ease !important;
    white-space: normal !important;
    text-align: left !important;
    line-height: 1.4 !important;
    min-height: 2.4rem !important;
    height: auto !important;
}
.stButton > button[kind="secondary"]:hover {
    background: #f5f3ef !important;
    border-color: #c8c2bb !important;
    color: #2d2a26 !important;
    transform: none !important;
}

[data-testid="stMetric"] {
    background: #ffffff !important; border: 1px solid #e8e4df !important;
    border-radius: 12px !important; padding: 0.9rem 1.1rem !important;
    box-shadow: 0 1px 4px rgba(45, 42, 38, 0.03) !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.72rem !important; font-weight: 500 !important; color: #9b9590 !important;
    text-transform: uppercase !important; letter-spacing: 0.06em !important;
}
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important; font-weight: 500 !important;
    font-size: 1.4rem !important; color: #2d2a26 !important;
}

.stTextArea textarea {
    border: 1.5px solid #e0dbd5 !important; border-radius: 12px !important;
    background: #ffffff !important; padding: 1rem !important; font-size: 0.95rem !important;
}
.stTextArea textarea:focus {
    border-color: #7ba7cc !important;
    box-shadow: 0 0 0 3px rgba(123, 167, 204, 0.12) !important;
}

details {
    border: 1px solid #e8e4df !important; border-radius: 12px !important;
    background: #ffffff !important;
}

/* === PILLS === */
.pill {
    display: inline-block; padding: 0.25rem 0.75rem; border-radius: 20px;
    font-size: 0.78rem; font-weight: 500; margin-right: 0.4rem;
}
.pill-blue   { background: #e8f0f7; color: #5a8bb5; border: 1px solid #d0e2f0; }
.pill-purple { background: #f0edf8; color: #7b6eac; border: 1px solid #ddd6f0; }
.pill-green  { background: #eaf5ee; color: #5d8b6e; border: 1px solid #d0e8d8; }

.v-pass {
    display: inline-block; padding: 0.2rem 0.65rem; border-radius: 20px;
    font-size: 0.76rem; font-weight: 500; background: #eaf5ee; color: #5d8b6e;
    border: 1px solid #d0e8d8; margin: 0.15rem;
}
.v-fail {
    display: inline-block; padding: 0.2rem 0.65rem; border-radius: 20px;
    font-size: 0.76rem; font-weight: 500; background: #f8eded; color: #a06b6b;
    border: 1px solid #ecd8d8; margin: 0.15rem;
}

/* === SOURCE CARDS === */
.source-card {
    background: #ffffff; border: 1px solid #e8e4df; border-left: 4px solid #7ba7cc;
    border-radius: 10px; padding: 1.2rem 1.4rem; margin: 0.5rem 0 1rem 0;
    box-shadow: 0 1px 3px rgba(45,42,38,0.04);
}
.source-header {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 0.8rem; padding-bottom: 0.7rem;
    border-bottom: 1px solid #f0ede8; flex-wrap: wrap; gap: 0.5rem;
}
.source-meta { display: flex; gap: 0.5rem; flex-wrap: wrap; align-items: center; }
.source-tag {
    display: inline-block; font-size: 0.72rem; padding: 0.18rem 0.6rem;
    border-radius: 14px; font-weight: 500;
    background: #f5f3ef; color: #6b6560; border: 1px solid #e8e4df;
}
.source-tag-company { background: #e8f0f7; color: #5a8bb5; border-color: #d0e2f0; }
.source-tag-year    { background: #f0edf8; color: #7b6eac; border-color: #ddd6f0; }
.source-tag-section { background: #eaf5ee; color: #5d8b6e; border-color: #d0e8d8; }
.source-tag-sim     { background: #faf0e8; color: #b5784a; border-color: #f0dcc8;
                      font-family: 'JetBrains Mono', monospace; }
.source-link {
    display: inline-flex; align-items: center; gap: 0.3rem;
    font-size: 0.78rem; color: #5a8bb5; text-decoration: none; font-weight: 500;
    padding: 0.18rem 0.7rem; background: #f0f6fb;
    border-radius: 14px; border: 1px solid #d6e3ef;
    transition: all 0.2s ease;
}
.source-link:hover { background: #e3eef7; color: #3d6a8c; text-decoration: none; }
.source-text-quote {
    background: #faf8f5; border-left: 3px solid #d4c8b8;
    padding: 0.9rem 1.1rem; border-radius: 6px;
    font-style: italic; color: #5a554e;
    margin-top: 0.5rem; line-height: 1.7; white-space: pre-wrap; word-wrap: break-word;
}
.highlight {
    background: #fff3d6; padding: 0.05rem 0.25rem;
    border-radius: 3px; font-weight: 500;
}

.hero-sub {
    color: #6b6560; font-size: 1rem; line-height: 1.75;
    margin-top: -0.3rem; margin-bottom: 1.5rem;
}

/* Detected-context banner */
.detected-banner {
    background: linear-gradient(135deg, #f0f6fb 0%, #f0edf8 100%);
    border: 1px solid #d6e3ef;
    border-radius: 12px;
    padding: 0.85rem 1.1rem;
    margin: 1rem 0 1.5rem 0;
    font-size: 0.88rem;
    color: #4a5d72;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.detected-banner b { color: #2d2a26; }
.detected-empty {
    background: #fef9ed;
    border: 1px solid #f0e0b8;
    color: #8a6d2f;
}

.examples-label {
    font-size: 0.78rem; color: #9b9590;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.5rem;
    margin-top: 0.5rem;
}

.sidebar-tip {
    font-size: 0.85rem; color: #6b6560; line-height: 1.65;
    padding: 0.9rem 1rem; background: rgba(123,167,204,0.06);
    border-radius: 10px; border: 1px solid rgba(123,167,204,0.12);
}
.sidebar-tip b { color: #2d2a26; }
.sidebar-tip code {
    background: #ffffff;
    padding: 0.1rem 0.35rem;
    border-radius: 4px;
    font-size: 0.78rem;
    color: #5a8bb5;
    border: 1px solid #e0dbd5;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================
# CONSTANTS
# ============================================================

COMPANY_DISPLAY = {
    "AAPL":  "🍎 Apple",
    "AMZN":  "📦 Amazon",
    "GOOGL": "🔍 Google",
    "MSFT":  "💻 Microsoft",
    "TSLA":  "⚡ Tesla",
}

EXAMPLE_QUERIES = [
    "What were Apple's main risk factors in 2024?",
    "Compare Tesla's revenue between 2023 and 2024",
    "Compare Microsoft and Google cloud strategy",
    "Summarize Amazon's business outlook",
]


# ============================================================
# CITATION + HIGHLIGHT HELPERS
# ============================================================

CITATION_RE = re.compile(
    r"[\(\[]?\s*source\s*(\d+)\s*[\)\]]?",
    re.IGNORECASE
)


def style_citations(answer: str) -> str:
    def replacer(m):
        return f":orange[(Source {m.group(1)})]"
    return CITATION_RE.sub(replacer, answer)


def extract_keywords(question: str) -> list:
    stopwords = {
        "what", "were", "was", "the", "of", "in", "for", "are", "is", "and",
        "or", "a", "an", "to", "from", "on", "at", "with", "by", "did", "do",
        "does", "how", "why", "when", "where", "which", "this", "that",
        "these", "those", "compare", "between", "tell", "me", "about",
        "describe", "explain", "summary", "summarize",
        "apple", "amazon", "google", "alphabet", "microsoft", "tesla",
        "aapl", "amzn", "googl", "msft", "tsla"
    }
    words = re.findall(r"\b\w{4,}\b", question.lower())
    return [w for w in words if w not in stopwords]


def highlight_keywords(text: str, keywords: list) -> str:
    if not keywords:
        return text
    for kw in sorted(set(keywords), key=len, reverse=True):
        text = re.sub(
            rf"(?i)\b({re.escape(kw)})\b",
            r'<span class="highlight">\1</span>',
            text
        )
    return text


def html_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ============================================================
# SOURCE RENDERING
# ============================================================

def render_structured_source(idx, doc, meta, distance, query):
    similarity = round((1 - distance) * 100, 2)
    company = meta.get("company", "—")
    year = meta.get("year", "—")
    section = meta.get("section", "—")
    filing_id = meta.get("filing_id", "")

    sec_url = get_filing_url(company, filing_id) if filing_id else ""

    doc_text = doc[:1500] + ("..." if len(doc) > 1500 else "")
    escaped = html_escape(doc_text)
    keywords = extract_keywords(query)
    highlighted = highlight_keywords(escaped, keywords)

    link_html = (
        f'<a class="source-link" href="{sec_url}" target="_blank" '
        f'title="Open original 10-K filing on SEC EDGAR">🔗 View on SEC EDGAR</a>'
        if sec_url else ""
    )

    html = f"""
    <div class="source-card">
      <div class="source-header">
        <div class="source-meta">
          <span class="source-tag" style="font-weight:600;">Source {idx}</span>
          <span class="source-tag source-tag-company">🏢 {company}</span>
          <span class="source-tag source-tag-year">📅 {year}</span>
          <span class="source-tag source-tag-section">📂 {section}</span>
          <span class="source-tag source-tag-sim">{similarity}%</span>
        </div>
        {link_html}
      </div>
      <div class="source-text-quote">{highlighted}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def source_cards(results, title, query):
    st.markdown(f"#### {title}")
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    if not docs:
        st.info("No sources retrieved.")
        return

    for i in range(len(docs)):
        render_structured_source(i + 1, docs[i], metas[i], dists[i], query)


def validation_badges(v):
    if not v:
        return
    st.markdown("#### Validation")
    checks = [
        ("Citations", v.get("citations_present", False)),
        ("Grounded", not v.get("unsupported_claim_detected", False)),
        ("Year Coverage", v.get("years_coverage_ok", True)),
        ("Numeric", v.get("numeric_present", True)),
        ("Overall", v.get("overall_valid", False)),
    ]
    html = " ".join([
        f"<span class='{'v-pass' if ok else 'v-fail'}'>{'✓' if ok else '✗'} {lbl}</span>"
        for lbl, ok in checks
    ])
    st.markdown(html, unsafe_allow_html=True)
    for w in v.get("warnings", []):
        st.warning(w)


# ============================================================
# DETECTION BANNER
# ============================================================

def render_detected_banner(companies, years):
    """Show what was auto-detected from the query."""
    parts = []

    if companies:
        display = " · ".join(COMPANY_DISPLAY.get(c, c) for c in companies)
        parts.append(f"<b>Company:</b> {display}")
    else:
        parts.append("<b>Company:</b> all available filings")

    if years:
        parts.append(f"<b>Year:</b> {' · '.join(str(y) for y in years)}")
    else:
        parts.append("<b>Year:</b> latest available")

    banner_class = "detected-banner"
    if not companies:
        banner_class += " detected-empty"

    st.markdown(
        f"<div class='{banner_class}'>"
        f"🪄 &nbsp; {' &nbsp;·&nbsp; '.join(parts)}"
        f"</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# SIDEBAR — minimal, just pipeline + help
# ============================================================

with st.sidebar:
    st.markdown("## Settings")

    pipeline_type = st.selectbox(
        "Pipeline",
        ["Agentic RAG", "Baseline RAG", "Compare Both"],
        help="Agentic uses query classification, section routing, and validation. "
             "Baseline uses static retrieval. Compare Both runs them side-by-side."
    )

    st.markdown("---")
    st.markdown(
        "<div class='sidebar-tip'>"
        "💡 <b>Just type your question.</b><br><br>"
        "Mention any company by name (<code>Apple</code>, <code>Tesla</code>, etc.) "
        "and the system will detect it automatically. Same with years."
        "<br><br>"
        "Available filings: <b>Apple, Amazon, Google, Microsoft, Tesla</b> "
        "(2022–2025)."
        "</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# HEADER
# ============================================================

st.markdown("# 📊 Financial Agentic RAG")
st.markdown(
    "<p class='hero-sub'>"
    "Ask questions about SEC 10-K filings in natural language. "
    "Company and year are detected automatically from your query."
    "</p>",
    unsafe_allow_html=True
)


# ============================================================
# QUERY INPUT
# ============================================================

# Pre-populate from a clicked example BEFORE the widget renders
if "pending_question" in st.session_state:
    st.session_state.question_input = st.session_state.pending_question
    del st.session_state.pending_question

if "question_input" not in st.session_state:
    st.session_state.question_input = ""

question = st.text_area(
    "Ask a question about SEC filings",
    height=100,
    key="question_input",
    placeholder="e.g. Compare Tesla's revenue with Apple's in 2024",
)

# Example query chips — 2 per row so text doesn't overflow
st.markdown(
    "<div class='examples-label'>Try an example</div>",
    unsafe_allow_html=True,
)

for row_start in range(0, len(EXAMPLE_QUERIES), 2):
    chip_cols = st.columns(2)
    for j, example in enumerate(EXAMPLE_QUERIES[row_start:row_start + 2]):
        with chip_cols[j]:
            if st.button(example, key=f"ex_{row_start + j}", use_container_width=True):
                # Stash the example, then rerun. The block at the top
                # will copy it into question_input BEFORE the widget exists.
                st.session_state.pending_question = example
                st.rerun()

st.markdown("")

# Analyze button
col_btn, _ = st.columns([1, 4])
with col_btn:
    run = st.button("Analyze", type="primary", use_container_width=True)


# ============================================================
# MAIN
# ============================================================

if run:
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        try:
            detected_companies = extract_companies(question)
            detected_years = extract_years(question)

            with st.spinner("Analyzing SEC filings …"):
                qtype = classify_query(question)
                strategy = build_retrieval_strategy(question)
                sec = strategy.get("section")

                if pipeline_type == "Baseline RAG":
                    t0 = time.time()
                    res = baseline_answer(
                        question=question, company=None,
                        year=None, section=sec
                    )
                    dt = round(time.time() - t0, 2)

                elif pipeline_type == "Agentic RAG":
                    t0 = time.time()
                    res = agentic_answer(question=question, company=None, year=None)
                    dt = round(time.time() - t0, 2)

                else:
                    t0 = time.time()
                    res_b = baseline_answer(
                        question=question, company=None,
                        year=None, section=sec
                    )
                    dt_b = round(time.time() - t0, 2)
                    t0 = time.time()
                    res_a = agentic_answer(question=question, company=None, year=None)
                    dt_a = round(time.time() - t0, 2)

            render_detected_banner(detected_companies, detected_years)

            st.markdown(
                f"<span class='pill pill-blue'>{pipeline_type}</span>"
                f"<span class='pill pill-purple'>{qtype}</span>"
                f"<span class='pill pill-green'>{sec or 'multi-section'}</span>",
                unsafe_allow_html=True,
            )

            with st.expander("Retrieval Strategy"):
                st.json(strategy)
            st.markdown("")

            # SINGLE PIPELINE
            if pipeline_type in ("Baseline RAG", "Agentic RAG"):
                st.markdown("### Answer")
                st.markdown(style_citations(res["answer"]))
                st.markdown("")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Response Time", f"{dt}s")
                m2.metric("Avg Similarity", f"{res['avg_similarity']}%")
                m3.metric("Top Similarity", f"{res['top_similarity']}%")
                m4.metric("Chunks", res["retrieved_chunks"])

                if res.get("retry_used"):
                    st.warning("Retry was triggered.")
                else:
                    st.success("No retry needed.")

                if pipeline_type == "Agentic RAG" and res.get("validation"):
                    validation_badges(res["validation"])

                st.markdown("")
                source_cards(res["results"], "Retrieved Sources", question)

            # COMPARE BOTH
            else:
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### Baseline RAG")
                    st.markdown(style_citations(res_b["answer"]))
                    st.markdown("")
                    b1, b2 = st.columns(2)
                    b1.metric("Time", f"{dt_b}s")
                    b2.metric("Avg Sim", f"{res_b['avg_similarity']}%")
                    b3, b4 = st.columns(2)
                    b3.metric("Top Sim", f"{res_b['top_similarity']}%")
                    b4.metric("Chunks", res_b["retrieved_chunks"])
                    if res_b.get("retry_used"):
                        st.warning("Retry triggered.")
                    else:
                        st.success("No retry.")

                with col2:
                    st.markdown("#### Agentic RAG")
                    st.markdown(style_citations(res_a["answer"]))
                    st.markdown("")
                    a1, a2 = st.columns(2)
                    a1.metric("Time", f"{dt_a}s")
                    a2.metric("Avg Sim", f"{res_a['avg_similarity']}%")
                    a3, a4 = st.columns(2)
                    a3.metric("Top Sim", f"{res_a['top_similarity']}%")
                    a4.metric("Chunks", res_a["retrieved_chunks"])
                    if res_a.get("retry_used"):
                        st.warning("Retry triggered.")
                    else:
                        st.success("No retry.")
                    if res_a.get("validation"):
                        validation_badges(res_a["validation"])

                st.markdown("")
                t1, t2 = st.tabs(["Baseline Sources", "Agentic Sources"])
                with t1:
                    source_cards(res_b["results"], "Baseline Sources", question)
                with t2:
                    source_cards(res_a["results"], "Agentic Sources", question)

        except Exception as e:
            st.error("An error occurred.")
            st.exception(e)