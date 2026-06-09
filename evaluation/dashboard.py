"""
evaluation/dashboard.py

Streamlit evaluation dashboard.

Run:
    streamlit run evaluation/dashboard.py

WHAT'S NEW (additive — all original charts kept)
------------------------------------------------
The dashboard previously visualised only the boolean columns in
results.csv (citation/section/similarity). It now also surfaces the three
advanced evaluators that already existed in the repo but were never shown:

1. TRUE NUMERIC ACCURACY (numeric_grader.matches): for numeric/comparative
   questions with a numeric gold, checks whether the generated answer
   actually contains the right figure within tolerance — not merely whether
   *some* number is present.
2. LLM-AS-JUDGE (results_judged.csv -> judge_score): average semantic score
   and a score distribution for prose (risk/descriptive) answers. The
   dashboard auto-loads results_judged.csv if present, else results.csv.
3. SYNONYM-NORMALIZED GOLD OVERLAP (metrics_synonym_patch.gold_overlap_metric):
   token recall of gold terms after collapsing financial synonyms.

Each advanced block is wrapped in try/except so the dashboard still loads if
a module/file is missing.
"""

import sys
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import load_results, compute_all_metrics

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_CSV = EVAL_DIR / "results.csv"
JUDGED_CSV = EVAL_DIR / "results_judged.csv"

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="RAG Evaluation Dashboard",
    layout="wide",
    page_icon="📊"
)

# ============================================================
# CUSTOM CSS — same palette as app.py
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=JetBrains+Mono:wght@400;500&display=swap');

.stApp {
    background-color: #faf8f5 !important;
    font-family: 'DM Sans', sans-serif !important;
}

header[data-testid="stHeader"] {
    background: transparent !important;
}

.block-container {
    padding-top: 2rem !important;
    max-width: 1300px !important;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f0ede8 0%, #e8e4dd 100%) !important;
    border-right: 1px solid #ddd8d0 !important;
}

h1 {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 700 !important;
    color: #2d2a26 !important;
    letter-spacing: -0.03em !important;
    font-size: 2rem !important;
}

h2, h3, h4 {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    color: #2d2a26 !important;
}

p, li, span, div, td, th {
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stMetric"] {
    background: #ffffff !important;
    border: 1px solid #e8e4df !important;
    border-radius: 12px !important;
    padding: 0.9rem 1.1rem !important;
    box-shadow: 0 1px 4px rgba(45,42,38,0.03) !important;
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

details {
    border: 1px solid #e8e4df !important;
    border-radius: 12px !important;
    background: #ffffff !important;
}

hr { border-color: #e8e4df !important; opacity: 0.6 !important; }

.hero-sub {
    color: #6b6560;
    font-size: 1rem;
    line-height: 1.7;
    margin-top: -0.3rem;
    margin-bottom: 1.5rem;
}

.section-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #9b9590;
    margin-bottom: 0.6rem;
    margin-top: 1.5rem;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# PLOTLY THEME COLORS
# ============================================================

CLR_BASELINE = "#7ba7cc"
CLR_AGENTIC = "#d4956a"
CLR_GREEN = "#7dab8e"
CLR_BG = "#faf8f5"
CLR_GRID = "#e8e4df"

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#2d2a26"),
    margin=dict(l=40, r=20, t=30, b=40),
    xaxis=dict(gridcolor=CLR_GRID, showgrid=False),
    yaxis=dict(gridcolor=CLR_GRID, gridwidth=1),
)


def styled_bar(df, x, y, color, color_map, text_auto=".1f", **kwargs):
    fig = px.bar(df, x=x, y=y, color=color, color_discrete_map=color_map,
                 text_auto=text_auto, **kwargs)
    fig.update_layout(**PLOTLY_LAYOUT, showlegend=False, height=330)
    fig.update_traces(marker_line_width=0)
    return fig


# ============================================================
# LOAD DATA — prefer the judged CSV (adds judge_score) if present
# ============================================================

@st.cache_data
def load_data():
    if JUDGED_CSV.exists():
        return load_results(JUDGED_CSV), True
    if RESULTS_CSV.exists():
        return load_results(RESULTS_CSV), False
    return None, False

df, judged_loaded = load_data()

if df is None:
    st.error(f"No results at `{RESULTS_CSV}`. Run `python -m evaluation.run_evaluation` first.")
    st.stop()

metrics = compute_all_metrics(df)

# ============================================================
# HEADER
# ============================================================

st.markdown("# 📊 Evaluation Dashboard")
st.markdown(
    "<p class='hero-sub'>Automated metrics comparing <b>Baseline RAG</b> and <b>Agentic RAG</b> "
    "across 40 SEC filing questions.</p>",
    unsafe_allow_html=True,
)

# ============================================================
# SIDEBAR FILTERS
# ============================================================

with st.sidebar:
    st.markdown("## Filters")
    pipelines = df["pipeline"].unique().tolist()
    sel_pipes = st.multiselect("Pipeline", pipelines, default=pipelines)
    atypes = df["answer_type"].unique().tolist()
    sel_types = st.multiselect("Answer Type", atypes, default=atypes)
    st.markdown("---")
    st.caption(
        "Source: " + ("results_judged.csv (with LLM-judge scores)"
                      if judged_loaded else "results.csv "
                      "(run `python -m evaluation.llm_judge` to add judge scores)")
    )

fdf = df[df["pipeline"].isin(sel_pipes) & df["answer_type"].isin(sel_types)]

# ============================================================
# KEY METRICS
# ============================================================

st.markdown("<div class='section-label'>Key Metrics</div>", unsafe_allow_html=True)

cols = st.columns(4)
for i, pipe in enumerate(["baseline", "agentic"]):
    pdf = fdf[fdf["pipeline"] == pipe]
    if pdf.empty:
        continue
    label = "Baseline" if pipe == "baseline" else "Agentic"
    cols[i * 2].metric(f"{label} Latency", f"{pdf['response_time_sec'].mean():.2f}s")
    cols[i * 2 + 1].metric(f"{label} Citation Rate", f"{pdf['citations_present'].mean()*100:.0f}%")

cols2 = st.columns(4)
for i, pipe in enumerate(["baseline", "agentic"]):
    pdf = fdf[fdf["pipeline"] == pipe]
    if pdf.empty:
        continue
    label = "Baseline" if pipe == "baseline" else "Agentic"
    cols2[i * 2].metric(f"{label} Section Acc.", f"{pdf['section_accuracy'].mean()*100:.0f}%")
    cols2[i * 2 + 1].metric(f"{label} Avg Similarity", f"{pdf['avg_similarity'].mean():.1f}%")

st.markdown("---")

# ============================================================
# CHARTS ROW 1
# ============================================================

st.markdown("<div class='section-label'>Pipeline Comparison</div>", unsafe_allow_html=True)

c1, c2 = st.columns(2)

cmap = {"baseline": CLR_BASELINE, "agentic": CLR_AGENTIC}

with c1:
    st.markdown("**Response Time**")
    lat = fdf.groupby("pipeline")["response_time_sec"].mean().reset_index()
    lat.columns = ["Pipeline", "Avg Latency (s)"]
    fig = styled_bar(lat, "Pipeline", "Avg Latency (s)", "Pipeline", cmap, text_auto=".2f")
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.markdown("**Retrieval Similarity**")
    sim = fdf.groupby("pipeline")[["avg_similarity", "top_similarity"]].mean().reset_index()
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(name="Avg", x=sim["pipeline"], y=sim["avg_similarity"],
                          marker_color=CLR_BASELINE, text=sim["avg_similarity"].round(1),
                          textposition="outside"))
    fig2.add_trace(go.Bar(name="Top", x=sim["pipeline"], y=sim["top_similarity"],
                          marker_color=CLR_GREEN, text=sim["top_similarity"].round(1),
                          textposition="outside"))
    fig2.update_layout(**PLOTLY_LAYOUT, barmode="group", height=330,
                       yaxis_title="Similarity (%)",
                       legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"))
    st.plotly_chart(fig2, use_container_width=True)

# ============================================================
# CHARTS ROW 2
# ============================================================

c3, c4 = st.columns(2)

with c3:
    st.markdown("**Retry Frequency** (Agentic)")
    adf = fdf[fdf["pipeline"] == "agentic"]
    if not adf.empty:
        rc = adf["retry_used"].value_counts().reset_index()
        rc.columns = ["Retry", "Count"]
        rc["Retry"] = rc["Retry"].map({True: "Retry", False: "No Retry"})
        fig3 = px.pie(rc, names="Retry", values="Count",
                      color_discrete_sequence=[CLR_AGENTIC, CLR_BASELINE],
                      hole=0.45)
        fig3.update_layout(**PLOTLY_LAYOUT, height=330, showlegend=True,
                           legend=dict(orientation="h", y=-0.05, x=0.5, xanchor="center"))
        fig3.update_traces(textinfo="percent+label", textfont_size=12)
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No agentic data.")

with c4:
    st.markdown("**Section Accuracy by Type**")
    acc = fdf.groupby(["pipeline", "answer_type"])["section_accuracy"].mean().reset_index()
    acc["section_accuracy"] = (acc["section_accuracy"] * 100).round(1)
    fig4 = px.bar(acc, x="answer_type", y="section_accuracy", color="pipeline",
                  barmode="group", color_discrete_map=cmap, text_auto=".0f",
                  labels={"section_accuracy": "Accuracy (%)", "answer_type": ""})
    fig4.update_layout(**PLOTLY_LAYOUT, height=330,
                       legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"))
    st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")

# ============================================================
# ANSWER QUALITY
# ============================================================

st.markdown("<div class='section-label'>Answer Quality</div>", unsafe_allow_html=True)

q1, q2 = st.columns(2)

with q1:
    st.markdown("**Citation Presence**")
    cit = fdf.groupby("pipeline")["citations_present"].mean().reset_index()
    cit["citations_present"] = (cit["citations_present"] * 100).round(1)
    fig5 = styled_bar(cit, "pipeline", "citations_present", "pipeline", cmap,
                      labels={"citations_present": "Rate (%)", "pipeline": ""})
    st.plotly_chart(fig5, use_container_width=True)

with q2:
    st.markdown("**Comparison Completeness**")
    comp = fdf[fdf["answer_type"] == "comparative"]
    if not comp.empty:
        ca = comp.groupby("pipeline")["years_coverage_ok"].mean().reset_index()
        ca["years_coverage_ok"] = (ca["years_coverage_ok"] * 100).round(1)
        fig6 = styled_bar(ca, "pipeline", "years_coverage_ok", "pipeline", cmap,
                          labels={"years_coverage_ok": "Both Years (%)", "pipeline": ""})
        st.plotly_chart(fig6, use_container_width=True)
    else:
        st.info("No comparative questions selected.")

st.markdown("---")

# ============================================================
# SEMANTIC & NUMERIC EVALUATION  (advanced — newly integrated)
# ============================================================

st.markdown("<div class='section-label'>Semantic &amp; Numeric Evaluation</div>",
            unsafe_allow_html=True)

# --- 1. True numeric accuracy (numeric_grader.matches) -----------------
try:
    from evaluation.numeric_grader import matches as numeric_matches
    from evaluation.numeric_grader import extract_values, TOLERANCE

    num_rows = fdf[fdf["answer_type"].isin(["numeric", "comparative"])]
    num_stats = {}
    for pipe, group in num_rows.groupby("pipeline"):
        correct = total = 0
        for _, r in group.iterrows():
            gold = str(r.get("gold_answer", ""))
            if not extract_values(gold):
                continue  # gold has no number -> graded by the judge instead
            total += 1
            correct += int(numeric_matches(gold, str(r.get("generated_answer", ""))))
        if total:
            num_stats[pipe] = dict(acc=correct / total * 100, correct=correct, total=total)

    na1, na2 = st.columns(2)
    with na1:
        st.markdown(f"**True Numeric Accuracy** (±{int(TOLERANCE*100)}% tolerance)")
        if num_stats:
            ndf = pd.DataFrame([
                {"pipeline": p, "accuracy": s["acc"]} for p, s in num_stats.items()
            ])
            fign = styled_bar(ndf, "pipeline", "accuracy", "pipeline", cmap,
                              labels={"accuracy": "Correct figure (%)", "pipeline": ""})
            st.plotly_chart(fign, use_container_width=True)
            st.caption("  ·  ".join(
                f"{p}: {s['correct']}/{s['total']}" for p, s in num_stats.items()
            ) + ". Counts only questions whose gold answer contains a figure; the "
                "answer must contain that figure within tolerance — not merely *a* number.")
        else:
            st.info("No numeric/comparative questions with numeric gold in selection.")

    # --- 2. Synonym-normalized gold overlap ----------------------------
    with na2:
        st.markdown("**Synonym-Normalized Gold Overlap**")
        try:
            from evaluation.metrics_synonym_patch import gold_overlap_metric
            ov = gold_overlap_metric(fdf)["per_pipeline"]
            odf = pd.DataFrame([
                {"pipeline": p, "overlap": v * 100} for p, v in ov.items()
            ])
            figo = styled_bar(odf, "pipeline", "overlap", "pipeline", cmap,
                              labels={"overlap": "Token recall (%)", "pipeline": ""})
            st.plotly_chart(figo, use_container_width=True)
            st.caption("Recall of gold-answer terms after collapsing financial "
                       "synonyms (revenue = net sales = total revenues, etc.).")
        except Exception as e:
            st.warning(f"Synonym overlap unavailable: {e}")
except Exception as e:
    st.warning(f"Numeric grading unavailable: {e}")

# --- 3. LLM-as-judge (prose answers) -----------------------------------
if judged_loaded and "judge_score" in df.columns:
    jdf = fdf.copy()
    jdf["judge_score"] = pd.to_numeric(jdf["judge_score"], errors="coerce")
    jdf = jdf.dropna(subset=["judge_score"])

    if not jdf.empty:
        j1, j2 = st.columns(2)
        with j1:
            st.markdown("**LLM Judge — Average Score** (risk + descriptive)")
            javg = jdf.groupby("pipeline")["judge_score"].mean().reset_index()
            javg["judge_score"] = javg["judge_score"].round(3)
            figj = styled_bar(javg, "pipeline", "judge_score", "pipeline", cmap,
                              text_auto=".2f",
                              labels={"judge_score": "Avg score (0–1)", "pipeline": ""})
            figj.update_yaxes(range=[0, 1])
            st.plotly_chart(figj, use_container_width=True)
        with j2:
            st.markdown("**LLM Judge — Score Distribution**")
            figd = px.histogram(jdf, x="judge_score", color="pipeline",
                                nbins=11, barmode="overlay", opacity=0.7,
                                color_discrete_map=cmap, range_x=[0, 1],
                                labels={"judge_score": "Judge score (0–1)"})
            figd.update_layout(**PLOTLY_LAYOUT, height=330,
                               legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"))
            st.plotly_chart(figd, use_container_width=True)
    else:
        st.info("No judge scores in the current selection.")
else:
    st.info("LLM-judge scores not loaded. Run `python -m evaluation.llm_judge` "
            "to generate results_judged.csv, then reload.")

st.markdown("---")

# ============================================================
# DATA TABLE
# ============================================================

st.markdown("<div class='section-label'>Detailed Results</div>", unsafe_allow_html=True)

show_cols = [c for c in [
    "pipeline", "question", "company", "year", "answer_type",
    "response_time_sec", "avg_similarity", "top_similarity",
    "section_accuracy", "citations_present", "retry_used", "judge_score",
] if c in fdf.columns]

st.dataframe(fdf[show_cols].reset_index(drop=True), use_container_width=True, height=380)

st.markdown("---")

# ============================================================
# INDIVIDUAL ANSWER REVIEW
# ============================================================

st.markdown("<div class='section-label'>Answer Review</div>", unsafe_allow_html=True)

q_opts = fdf["question"].unique().tolist()
sel_q = st.selectbox("Select a question", q_opts)

if sel_q:
    rows = fdf[fdf["question"] == sel_q]
    for _, row in rows.iterrows():
        pipe_label = "Baseline" if row["pipeline"] == "baseline" else "Agentic"
        with st.expander(
            f"{pipe_label}  ·  {row['response_time_sec']}s  ·  "
            f"Sim {row['avg_similarity']}%"
        ):
            st.markdown(f"**Gold:** {row.get('gold_answer', 'N/A')}")
            st.markdown(f"**Generated:**")
            st.write(row.get("generated_answer", "N/A"))
            if row.get("judge_score") not in (None, "", float("nan")):
                jr = row.get("judge_reason", "")
                st.markdown(f"**Judge:** {row.get('judge_score')}  —  {jr}")
            mc = st.columns(6)
            mc[0].metric("Sec. Acc.", "✓" if row.get("section_accuracy") else "✗")
            mc[1].metric("Citations", "✓" if row.get("citations_present") else "✗")
            mc[2].metric("Retry", "✓" if row.get("retry_used") else "✗")
            mc[3].metric("Years", "✓" if row.get("years_coverage_ok") else "✗")
            mc[4].metric("Numeric", "✓" if row.get("numeric_present") else "✗")
            mc[5].metric("Avg Sim", f"{row.get('avg_similarity', 0):.1f}%")

# ============================================================
# FULL METRICS JSON
# ============================================================

with st.expander("Full Metrics (JSON)"):
    st.json(metrics)