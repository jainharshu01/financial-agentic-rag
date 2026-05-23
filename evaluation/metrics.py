"""
evaluation/metrics.py

Compute evaluation metrics from results.csv produced by run_evaluation.py.

Metrics implemented:
- Hit@k               : whether expected_section appears in top-k retrieved sections
- Section Accuracy    : fraction of questions where correct section was retrieved
- Groundedness        : fraction of answers that contain citations
- Numeric Accuracy    : fraction of numeric-type answers containing a number
- Citation Presence   : fraction of all answers containing (Source X) citations
- Comparison Complete : fraction of comparative answers covering both years
- Average Latency     : mean response time per pipeline
- Retry Frequency     : fraction of agentic queries that triggered retry

Usage:
    from evaluation.metrics import compute_all_metrics, load_results
    df = load_results()
    report = compute_all_metrics(df)
"""

import re
import sys
from pathlib import Path

import pandas as pd

# ============================================================
# PATH FIX
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_CSV = EVAL_DIR / "results.csv"

CITATION_PATTERN = re.compile(r"\(source\s*\d+\)", re.IGNORECASE)
NUMBER_PATTERN = re.compile(
    r"\b\d[\d,\.]*\s*(?:billion|million|thousand|%|percent)?\b", re.IGNORECASE
)


# ============================================================
# LOAD RESULTS
# ============================================================

def load_results(path: Path = RESULTS_CSV) -> pd.DataFrame:
    """Load results CSV into a DataFrame."""
    df = pd.read_csv(path)
    # Coerce boolean columns that may be read as strings
    for col in ["retry_used", "section_accuracy", "citations_present",
                "years_coverage_ok", "numeric_present", "overall_valid"]:
        if col in df.columns:
            df[col] = df[col].map(
                lambda x: True if str(x).strip().lower() in ("true", "1", "yes") else False
            )
    return df


# ============================================================
# INDIVIDUAL METRICS
# ============================================================

def hit_at_k(df: pd.DataFrame, k: int = 3) -> dict:
    """
    Hit@k: proportion of questions where the expected section
    appears among the retrieved sections (retrieved_chunks >= 1 proxy).

    We use section_accuracy which flags whether the expected section
    was retrieved — this is effectively Hit@k since we retrieve top_k chunks.
    """
    per_pipeline = {}
    for pipeline, group in df.groupby("pipeline"):
        hit = group["section_accuracy"].sum()
        total = len(group)
        per_pipeline[pipeline] = round(hit / total, 4) if total > 0 else 0.0

    return {
        "metric": f"Hit@{k}",
        "description": "Fraction of queries where expected section was retrieved",
        "per_pipeline": per_pipeline,
    }


def section_accuracy_metric(df: pd.DataFrame) -> dict:
    """Fraction of queries with correct section retrieved."""
    per_pipeline = {}
    per_type = {}

    for pipeline, group in df.groupby("pipeline"):
        acc = group["section_accuracy"].mean()
        per_pipeline[pipeline] = round(acc, 4)

    for atype, group in df.groupby("answer_type"):
        acc = group["section_accuracy"].mean()
        per_type[atype] = round(acc, 4)

    return {
        "metric": "Section Accuracy",
        "description": "Fraction of queries retrieving the correct SEC section",
        "per_pipeline": per_pipeline,
        "per_answer_type": per_type,
    }


def groundedness_metric(df: pd.DataFrame) -> dict:
    """
    Groundedness: fraction of answers that contain at least one citation.
    Uses citations_present column (pre-computed in run_evaluation.py).
    """
    per_pipeline = {}
    for pipeline, group in df.groupby("pipeline"):
        score = group["citations_present"].mean()
        per_pipeline[pipeline] = round(score, 4)

    return {
        "metric": "Groundedness (Citation Presence)",
        "description": "Fraction of answers containing at least one (Source X) citation",
        "per_pipeline": per_pipeline,
    }


def numeric_accuracy_metric(df: pd.DataFrame) -> dict:
    """
    Numeric Accuracy: for numeric-type questions only,
    fraction of answers containing a numeric value.
    """
    numeric_df = df[df["answer_type"] == "numeric"]

    if numeric_df.empty:
        return {
            "metric": "Numeric Accuracy",
            "description": "Fraction of numeric-type answers containing a number",
            "per_pipeline": {},
            "note": "No numeric questions found in results.",
        }

    per_pipeline = {}
    for pipeline, group in numeric_df.groupby("pipeline"):
        score = group["numeric_present"].mean()
        per_pipeline[pipeline] = round(score, 4)

    return {
        "metric": "Numeric Accuracy",
        "description": "Fraction of numeric-type answers containing a number",
        "per_pipeline": per_pipeline,
    }


def citation_presence_metric(df: pd.DataFrame) -> dict:
    """
    Citation Presence: same as groundedness but broken out
    per answer type for finer granularity.
    """
    per_type = {}
    for atype, group in df.groupby("answer_type"):
        score = group["citations_present"].mean()
        per_type[atype] = round(score, 4)

    overall = round(df["citations_present"].mean(), 4)

    return {
        "metric": "Citation Presence",
        "description": "Fraction of answers containing citations, by answer type",
        "overall": overall,
        "per_answer_type": per_type,
    }


def comparison_completeness_metric(df: pd.DataFrame) -> dict:
    """
    Comparison Completeness: for comparative-type questions,
    fraction of answers that cover both years.
    """
    comp_df = df[df["answer_type"] == "comparative"]

    if comp_df.empty:
        return {
            "metric": "Comparison Completeness",
            "description": "Fraction of comparative answers mentioning both years",
            "per_pipeline": {},
            "note": "No comparative questions found in results.",
        }

    per_pipeline = {}
    for pipeline, group in comp_df.groupby("pipeline"):
        score = group["years_coverage_ok"].mean()
        per_pipeline[pipeline] = round(score, 4)

    return {
        "metric": "Comparison Completeness",
        "description": "Fraction of comparative answers mentioning both years",
        "per_pipeline": per_pipeline,
    }


def average_latency_metric(df: pd.DataFrame) -> dict:
    """Average response time in seconds per pipeline."""
    per_pipeline = {}
    for pipeline, group in df.groupby("pipeline"):
        avg = group["response_time_sec"].mean()
        per_pipeline[pipeline] = round(avg, 2)

    return {
        "metric": "Average Latency (seconds)",
        "description": "Mean response time per pipeline",
        "per_pipeline": per_pipeline,
    }


def retry_frequency_metric(df: pd.DataFrame) -> dict:
    """Fraction of agentic queries that triggered a retry."""
    agentic_df = df[df["pipeline"] == "agentic"]

    if agentic_df.empty:
        return {
            "metric": "Retry Frequency",
            "description": "Fraction of agentic queries triggering retry",
            "agentic": None,
            "note": "No agentic results found.",
        }

    freq = agentic_df["retry_used"].mean()
    return {
        "metric": "Retry Frequency",
        "description": "Fraction of agentic queries triggering retry",
        "agentic": round(freq, 4),
    }


def similarity_stats_metric(df: pd.DataFrame) -> dict:
    """Average and top similarity scores per pipeline."""
    per_pipeline = {}
    for pipeline, group in df.groupby("pipeline"):
        per_pipeline[pipeline] = {
            "avg_similarity_mean": round(group["avg_similarity"].mean(), 2),
            "top_similarity_mean": round(group["top_similarity"].mean(), 2),
        }

    return {
        "metric": "Retrieval Similarity",
        "description": "Mean avg and top similarity scores per pipeline",
        "per_pipeline": per_pipeline,
    }


# ============================================================
# COMPUTE ALL METRICS
# ============================================================

def compute_all_metrics(df: pd.DataFrame) -> dict:
    """
    Compute all metrics and return as a structured dict.

    Args:
        df: DataFrame loaded from results.csv

    Returns:
        dict mapping metric_name → metric result dict
    """
    metrics = {}

    metrics["hit_at_3"] = hit_at_k(df, k=3)
    metrics["section_accuracy"] = section_accuracy_metric(df)
    metrics["groundedness"] = groundedness_metric(df)
    metrics["numeric_accuracy"] = numeric_accuracy_metric(df)
    metrics["citation_presence"] = citation_presence_metric(df)
    metrics["comparison_completeness"] = comparison_completeness_metric(df)
    metrics["average_latency"] = average_latency_metric(df)
    metrics["retry_frequency"] = retry_frequency_metric(df)
    metrics["similarity_stats"] = similarity_stats_metric(df)

    return metrics


def print_metrics_report(metrics: dict):
    """Pretty-print the metrics report."""
    print("\n" + "="*70)
    print("EVALUATION METRICS REPORT")
    print("="*70)

    for key, m in metrics.items():
        print(f"\n{m['metric']}")
        print(f"  {m['description']}")
        for k, v in m.items():
            if k not in ("metric", "description"):
                print(f"  {k}: {v}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    if not RESULTS_CSV.exists():
        print(f"ERROR: {RESULTS_CSV} not found. Run run_evaluation.py first.")
        sys.exit(1)

    df = load_results()
    print(f"Loaded {len(df)} result rows.")

    metrics = compute_all_metrics(df)
    print_metrics_report(metrics)