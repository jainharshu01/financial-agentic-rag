"""
app.py

Financial Agentic RAG — Main Streamlit UI.
"""

import time
import re
import streamlit as st

from src.retrieval.baseline_rag import baseline_answer
from src.agent.agentic_rag import agentic_answer
from src.agent.query_classifier import classify_query
from src.agent.router import build_retrieval_strategy

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

/* === GLOBAL === */
.stApp {
    background-color: #faf8f5 !important;
    font-family: 'DM Sans', sans-serif !important;
}

header[data-testid="stHeader"] {
    background: transparent !important;
}

.block-container {
    padding-top: 2rem !important;
    max-width: 1200px !important;
}

/* === SIDEBAR === */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f0ede8 0%, #e8e4dd 100%) !important;
    border-right: 1px solid #ddd8d0 !important;
}

section[data-testid="stSidebar"] .stMarkdown h2 {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    color: #3d3a36 !important;
    letter-spacing: 0.03em !important;
    text-transform: uppercase !important;
    margin-bottom: 1rem !important;
}

/* === TYPOGRAPHY === */
h1 {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 700 !important;
    color: #2d2a26 !important;
    letter-spacing: -0.03em !important;
    font-size: 2.2rem !important;
    margin-bottom: 0 !important;
}

h2, h3, h4 {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    color: #2d2a26 !important;
}

p, li, span, div {
    font-family: 'DM Sans', sans-serif !important;
}

/* === PRIMARY BUTTON === */
.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #7ba7cc 0%, #9b8ec4 100%) !important;
    color: white !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.55rem 1.8rem !important;
    font-size: 0.92rem !important;
    letter-spacing: 0.02em !important;
    box-shadow: 0 2px 10px rgba(123, 167, 204, 0.25) !important;
    transition: all 0.25s ease !important;
}

.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {
    box-shadow: 0 4px 18px rgba(123, 167, 204, 0.35) !important;
    transform: translateY(-1px) !important;
}

/* === METRIC CARDS === */
[data-testid="stMetric"] {
    background: #ffffff !important;
    border: 1px solid #e8e4df !important;
    border-radius: 12px !important;
    padding: 0.9rem 1.1rem !important;
    box-shadow: 0 1px 4px rgba(45, 42, 38, 0.03) !important;
}

[data-testid="stMetricLabel"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    color: #9b9590 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}

[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 500 !important;
    font-size: 1.4rem !important;
    color: #2d2a26 !important;
}

/* === TEXT AREA === */
.stTextArea textarea {
    font-family: 'DM Sans', sans-serif !important;
    border: 1.5px solid #e0dbd5 !important;
    border-radius: 12px !important;
    background: #ffffff !important;
    padding: 1rem !important;
    font-size: 0.95rem !important;
    color: #2d2a26 !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}

.stTextArea textarea:focus {
    border-color: #7ba7cc !important;
    box-shadow: 0 0 0 3px rgba(123, 167, 204, 0.12) !important;
}

/* === EXPANDERS === */
details {
    border: 1px solid #e8e4df !important;
    border-radius: 12px !important;
    background: #ffffff !important;
}

details summary {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
}

/* === TABS === */
.stTabs [data-baseweb="tab-list"] {
    border-bottom: 2px solid #e8e4df !important;
    gap: 0 !important;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    padding: 0.6rem 1.3rem !important;
    color: #9b9590 !important;
}

.stTabs [aria-selected="true"] {
    color: #7ba7cc !important;
    border-bottom-color: #7ba7cc !important;
}

/* === SELECTBOX === */
.stSelectbox > div > div {
    border-radius: 10px !important;
    border: 1.5px solid #e0dbd5 !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* === DIVIDER === */
hr {
    border-color: #e8e4df !important;
    opacity: 0.6 !important;
}

/* === CUSTOM CLASSES === */
.hero-sub {
    color: #6b6560;
    font-size: 1rem;
    line-height: 1.75;
    margin-top: -0.3rem;
    margin-bottom: 1.5rem;
}

.pill {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 500;
    font-family: 'DM Sans', sans-serif;
    margin-right: 0.4rem;
}
.pill-blue { background: #e8f0f7; color: #5a8bb5; border: 1px solid #d0e2f0; }
.pill-purple { background: #f0edf8; color: #7b6eac; border: 1px solid #ddd6f0; }
.pill-green { background: #eaf5ee; color: #5d8b6e; border: 1px solid #d0e8d8; }
.pill-orange { background: #faf0e8; color: #b5784a; border: 1px solid #f0dcc8; }

.v-pass {
    display: inline-block;
    padding: 0.2rem 0.65rem;
    border-radius: 20px;
    font-size: 0.76rem;
    font-weight: 500;
    background: #eaf5ee;
    color: #5d8b6e;
    border: 1px solid #d0e8d8;
    margin: 0.15rem;
}
.v-fail {
    display: inline-block;
    padding: 0.2rem 0.65rem;
    border-radius: 20px;
    font-size: 0.76rem;
    font-weight: 500;
    background: #f8eded;
    color: #a06b6b;
    border: 1px solid #ecd8d8;
    margin: 0.15rem;
}

.chunk-box {
    background: #f5f3ef;
    padding: 1rem 1.2rem;
    border-radius: 10px;
    font-size: 0.84rem;
    line-height: 1.7;
    color: #6b6560;
    border-left: 3px solid #7ba7cc;
    margin-top: 0.5rem;
}

.answer-box {
    background: #ffffff;
    border: 1px solid #e8e4df;
    border-radius: 14px;
    padding: 1.5rem 1.8rem;
    box-shadow: 0 2px 10px rgba(45,42,38,0.04);
    margin: 0.8rem 0 1.2rem 0;
    line-height: 1.75;
}

.sidebar-tip {
    font-size: 0.82rem;
    color: #9b9590;
    line-height: 1.6;
    padding: 0.8rem;
    background: rgba(123,167,204,0.06);
    border-radius: 10px;
    border: 1px solid rgba(123,167,204,0.1);
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================
# HELPERS
# ============================================================

CITATION_RE = re.compile(r"\(Source (\d+)\)", re.IGNORECASE)


def style_citations(answer: str) -> str:
    def replacer(m):
        return f":orange[(Source {m.group(1)})]"
    return CITATION_RE.sub(replacer, answer)


def source_cards(results: dict, title: str = "Retrieved Sources"):
    st.markdown(f"#### {title}")
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    for i in range(len(docs)):
        meta = metas[i]
        sim = round((1 - dists[i]) * 100, 2)
        with st.expander(
            f"Source {i+1}  ·  {meta.get('company','')}  ·  "
            f"{meta.get('section','')}  ·  {sim}%"
        ):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Company", meta.get("company", ""))
            c2.metric("Year", meta.get("year", ""))
            c3.metric("Section", meta.get("section", ""))
            c4.metric("Similarity", f"{sim}%")
            st.markdown(f"<div class='chunk-box'>{docs[i]}</div>", unsafe_allow_html=True)


def validation_badges(v: dict):
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
# HEADER
# ============================================================

st.markdown("# 📊 Financial Agentic RAG")
st.markdown(
    "<p class='hero-sub'>"
    "Analyze SEC 10-K filings from Apple · Tesla · Microsoft · Amazon · Google "
    "with intelligent retrieval, source grounding, and answer validation."
    "</p>",
    unsafe_allow_html=True
)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## Settings")

    pipeline_type = st.selectbox(
        "Pipeline",
        ["Baseline RAG", "Agentic RAG", "Compare Both"],
    )

    COMPANY_LABELS = {
        "AAPL": "🍎  Apple",
        "AMZN": "📦  Amazon",
        "GOOGL": "🔍  Google",
        "MSFT": "💻  Microsoft",
        "TSLA": "⚡  Tesla",
    }

    company = st.selectbox(
        "Company",
        list(COMPANY_LABELS.keys()),
        format_func=lambda x: COMPANY_LABELS[x],
    )

    year = st.selectbox("Year", ["Auto", 2022, 2023, 2024, 2025])
    if year == "Auto":
        year = None

    st.markdown("---")
    st.markdown(
        "<div class='sidebar-tip'>"
        "💡 <b>Compare Both</b> runs baseline and agentic side-by-side "
        "on the same question so you can see the difference."
        "</div>",
        unsafe_allow_html=True,
    )

# ============================================================
# QUERY INPUT
# ============================================================

question = st.text_area(
    "Ask a question about SEC filings",
    height=100,
    placeholder="e.g. Compare Apple's risk factors between 2023 and 2024",
)

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
            with st.spinner("Analyzing SEC filings …"):
                qtype = classify_query(question)
                strategy = build_retrieval_strategy(question)
                sec = strategy.get("section")

                if pipeline_type == "Baseline RAG":
                    t0 = time.time()
                    res = baseline_answer(question=question, company=company, year=year, section=sec)
                    dt = round(time.time() - t0, 2)

                elif pipeline_type == "Agentic RAG":
                    t0 = time.time()
                    res = agentic_answer(question=question, company=company, year=year)
                    dt = round(time.time() - t0, 2)

                else:
                    t0 = time.time()
                    res_b = baseline_answer(question=question, company=company, year=year, section=sec)
                    dt_b = round(time.time() - t0, 2)
                    t0 = time.time()
                    res_a = agentic_answer(question=question, company=company, year=year)
                    dt_a = round(time.time() - t0, 2)

            # --- Badges ---
            st.markdown("")
            st.markdown(
                f"<span class='pill pill-blue'>{pipeline_type}</span>"
                f"<span class='pill pill-purple'>{qtype}</span>"
                f"<span class='pill pill-green'>{sec or 'Auto'}</span>",
                unsafe_allow_html=True,
            )
            with st.expander("Retrieval Strategy"):
                st.json(strategy)
            st.markdown("")

            # ====================================================
            # SINGLE PIPELINE
            # ====================================================

            if pipeline_type in ("Baseline RAG", "Agentic RAG"):
                st.markdown(
                    f"<div class='answer-box'>{style_citations(res['answer'])}</div>",
                    unsafe_allow_html=True,
                )
                # use st.markdown fallback for citation coloring
                st.markdown(style_citations(res["answer"]))

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
                source_cards(res["results"])

            # ====================================================
            # COMPARE BOTH
            # ====================================================

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
                    source_cards(res_b["results"], "Baseline Sources")
                with t2:
                    source_cards(res_a["results"], "Agentic Sources")

        except Exception as e:
            st.error("An error occurred.")
            st.exception(e)