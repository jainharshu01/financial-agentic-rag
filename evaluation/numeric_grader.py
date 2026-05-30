"""
evaluation/numeric_grader.py   (NEW)

Objective auto-grading for numeric & comparative answers.

THE PROBLEM WITH GOLD ANSWERS
-----------------------------
Your 40 gold answers fall into two kinds:
  * 22 contain hard numbers (all 10 numeric + most 8 comparative + a few
    others). These can be graded OBJECTIVELY — no human needed.
  * 18 are descriptive/risk prose ("Apple faces risks including global
    competition, supply chain concentration..."). These cannot be graded
    by exact match; they need soft keyword overlap (already provided by
    evaluation/metrics_synonym_patch.py) or occasional human spot-checks.

This module grades the NUMERIC kind automatically:
  1. Pull the dollar/percent figures from the gold answer.
  2. Pull the figures from the generated answer.
  3. Mark correct if the generated answer contains a figure within
     TOLERANCE of the gold figure (default 2% relative, to allow
     rounding like 391 vs 391.04 billion).

It reads results.csv (already written by run_evaluation.py), so NO
re-run is required — just run this after an evaluation.

USAGE
-----
    python -m evaluation.numeric_grader
"""

import csv
import re
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_CSV = EVAL_DIR / "results.csv"

TOLERANCE = 0.02  # 2% relative tolerance

# Multipliers so "391 billion" and "391,040 million" compare equal.
SCALE = {
    "billion": 1e9, "bn": 1e9, "b": 1e9,
    "million": 1e6, "mn": 1e6, "m": 1e6,
    "thousand": 1e3, "k": 1e3,
    "%": 0.01, "percent": 0.01,
}

_NUM_RE = re.compile(
    r"\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*"
    r"(billion|bn|b|million|mn|m|thousand|k|percent|%)?",
    re.IGNORECASE,
)


def extract_values(text: str):
    """
    Return a list of normalized numeric values found in text.
    A bare 4-digit number that looks like a year (2019-2026) is dropped
    so '2024' isn't mistaken for a financial figure.
    """
    vals = []
    for m in _NUM_RE.finditer(str(text)):
        raw = m.group(1).replace(",", "")
        unit = (m.group(2) or "").lower()
        try:
            num = float(raw)
        except ValueError:
            continue
        # Drop bare years.
        if not unit and 2015 <= num <= 2030 and num == int(num):
            continue
        mult = SCALE.get(unit, 1.0)
        vals.append(num * mult)
    return vals


def matches(gold_text: str, gen_text: str, tol: float = TOLERANCE) -> bool:
    """
    True if EVERY distinct gold figure has a generated figure within tol.
    (For comparative answers with two figures, both must be present.)
    """
    gold_vals = [v for v in extract_values(gold_text) if v != 0]
    gen_vals = extract_values(gen_text)
    if not gold_vals:
        return False  # not a numeric gold; grade elsewhere

    for gv in gold_vals:
        ok = any(abs(gv - xv) <= tol * abs(gv) for xv in gen_vals)
        if not ok:
            return False
    return True


def grade():
    if not RESULTS_CSV.exists():
        print(f"ERROR: {RESULTS_CSV} not found. Run run_evaluation first.")
        sys.exit(1)

    rows = list(csv.DictReader(open(RESULTS_CSV, encoding="utf-8")))

    # Per-pipeline numeric accuracy on questions whose gold has numbers.
    stats = {}
    detail = []
    for r in rows:
        if r["answer_type"] not in ("numeric", "comparative"):
            continue
        gold = r.get("gold_answer", "")
        if not extract_values(gold):
            continue
        ok = matches(gold, r.get("generated_answer", ""))
        p = r["pipeline"]
        stats.setdefault(p, {"correct": 0, "total": 0})
        stats[p]["total"] += 1
        stats[p]["correct"] += int(ok)
        detail.append((p, r["answer_type"], ok, r["question"][:55]))

    print("\n" + "=" * 64)
    print("NUMERIC AUTO-GRADE (objective, tolerance = "
          f"{int(TOLERANCE*100)}%)")
    print("=" * 64)
    for p, s in stats.items():
        acc = s["correct"] / s["total"] * 100 if s["total"] else 0
        print(f"  {p:9s}: {s['correct']}/{s['total']} correct  "
              f"({acc:.1f}%)")

    print("\nPer-question:")
    for p, t, ok, q in sorted(detail):
        mark = "OK " if ok else "XX "
        print(f"  [{mark}] {p:9s} {t:11s} {q}")

    return stats


if __name__ == "__main__":
    grade()