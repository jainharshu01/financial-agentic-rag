"""
evaluation/metrics_synonym_patch.py   (NEW — small add-on)

Adds a synonym-aware answer-quality metric WITHOUT rewriting metrics.py.

WHY SEPARATE FILE
-----------------
The existing metrics.py only aggregates pre-computed boolean columns from
results.csv; it never compares generated answer text to the gold answer.
This patch adds a real token-overlap metric that normalizes financial
synonyms first, so "net sales" is credited against a gold "revenue".

USAGE
-----
    from evaluation.metrics import load_results
    from evaluation.metrics_synonym_patch import gold_overlap_metric

    df = load_results()
    print(gold_overlap_metric(df))

It reads the `generated_answer` and `gold_answer` columns that
run_evaluation.py already writes, so no re-run is required.
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.financial_synonyms import normalize_for_eval

_STOP = {
    "the", "a", "an", "of", "in", "for", "and", "or", "to", "from",
    "on", "at", "with", "by", "is", "was", "were", "this", "that",
    "source", "sources",
}


def _tokens(text: str) -> set:
    text = normalize_for_eval(str(text))
    words = re.findall(r"[a-z0-9$%\.]+", text.lower())
    return {w for w in words if w not in _STOP and len(w) > 1}


def gold_overlap_metric(df) -> dict:
    """
    Token-level recall of gold-answer terms in the generated answer,
    after financial-synonym normalization. Reported per pipeline.
    """
    per_pipeline = {}
    for pipeline, group in df.groupby("pipeline"):
        recalls = []
        for _, row in group.iterrows():
            gold = _tokens(row.get("gold_answer", ""))
            gen = _tokens(row.get("generated_answer", ""))
            if not gold:
                continue
            recalls.append(len(gold & gen) / len(gold))
        per_pipeline[pipeline] = (
            round(sum(recalls) / len(recalls), 4) if recalls else 0.0
        )

    return {
        "metric": "Synonym-Normalized Gold Overlap",
        "description": (
            "Token recall of gold-answer terms in the generated answer, "
            "after collapsing financial synonyms (revenue=net sales, etc.)"
        ),
        "per_pipeline": per_pipeline,
    }


if __name__ == "__main__":
    from evaluation.metrics import load_results
    df = load_results()
    print(gold_overlap_metric(df))