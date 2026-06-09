"""
evaluation/run_evaluation.py

Automated evaluation pipeline.
- Loads evaluation_questions.csv
- Runs EVERY question through baseline_rag and agentic_rag (both 70B)
- Stores results to evaluation/results.csv
- Then runs the NUMERIC GRADER (free, no API) and prints its results.

The LLM judge is a SEPARATE step:  python -m evaluation.llm_judge

FIXES / FEATURES IN THIS VERSION
--------------------------------
1. CRASH FIX (None.get). When agentic retrieval returned 0 chunks, the
   pipeline returned validation=None, and reading response.get("validation",
   {}) yielded None (the key existed), so the next .get() crashed with
   "'NoneType' object has no attribute 'get'". Now read with `or {}` so both
   the None and missing cases fall back to an empty dict. A 0-chunk question
   now records a normal row ("No relevant documents found.") instead of an
   error row.

2. --resume. Re-running with --resume loads the existing results.csv, KEEPS
   every row that previously SUCCEEDED, and only re-runs the (pipeline,
   question) pairs that errored or are missing. This is the cure for a run
   that died partway through on Groq 429 rate limits: re-run on a fresh-quota
   day (ideally with a bigger --sleep) and it fills in only the failed rows.

RATE-LIMIT HANDLING
-------------------
  --sleep : pause after every pipeline call (default 1.0s). The comparative
    questions send large prompts; if the tail of a run dies on 429s, raise
    this (e.g. --sleep 6) and/or use --resume. NOTE: backoff handles
    per-MINUTE limits; a per-DAY token cap is a hard wall — use --resume the
    next day.
  429 backoff: each pipeline call retries ONLY on rate-limit errors with
    exponential waits, so a throttle becomes a wait, not an error row.

Run:
    python -m evaluation.run_evaluation                  # fresh full run + numeric grade
    python -m evaluation.run_evaluation --resume         # retry only failed/missing rows
    python -m evaluation.run_evaluation --resume --sleep 6   # ...and go gentler on the API
    python -m evaluation.run_evaluation --limit 4        # smoke test, first 4 questions
    python -m evaluation.run_evaluation --no-grade       # answers only, skip grader
"""

import os
import sys
import csv
import time
import re
import argparse
import traceback
from pathlib import Path

# ============================================================
# PATH FIX — ensure project root is on sys.path
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.baseline_rag import baseline_answer
from src.agent.agentic_rag import agentic_answer
from src.agent.query_classifier import classify_query
from src.agent.router import build_retrieval_strategy

# ============================================================
# PATHS
# ============================================================

EVAL_DIR = Path(__file__).resolve().parent
QUESTIONS_CSV = EVAL_DIR / "evaluation_questions.csv"
RESULTS_CSV = EVAL_DIR / "results.csv"

# ============================================================
# RATE-LIMIT DEFAULTS  (overridable via CLI)
# ============================================================

DEFAULT_SLEEP = 1.0        # seconds to pause after each pipeline call
DEFAULT_MAX_RETRIES = 5    # backoff attempts on a 429 before giving up
DEFAULT_BASE_BACKOFF = 2.0  # first backoff wait; doubles each retry

# ============================================================
# OUTPUT SCHEMA
# ============================================================

FIELDNAMES = [
    "pipeline", "question", "company", "year", "answer_type",
    "expected_section", "gold_answer", "generated_answer",
    "response_time_sec", "retry_used", "avg_similarity", "top_similarity",
    "retrieved_chunks", "retrieved_sections", "section_accuracy",
    "citations_present", "citation_count",
    "years_coverage_ok", "numeric_present",
    "overall_valid", "error",
]

# ============================================================
# HELPERS
# ============================================================

# FLEXIBLE CITATION REGEX
# Matches: (Source 1), [Source 2], Source 3, (source 4), [source 5], source 6
CITATION_PATTERN = re.compile(
    r"[\(\[]?\s*source\s*\d+\s*[\)\]]?",
    re.IGNORECASE
)


def citations_present(answer: str) -> bool:
    """Return True if any citation format is present."""
    return bool(CITATION_PATTERN.search(answer))


def count_citations(answer: str) -> int:
    """Count total citation occurrences."""
    return len(CITATION_PATTERN.findall(answer))


def extract_sections_from_results(results: dict) -> list:
    """Return list of unique section names from retrieved chunks."""
    sections = set()
    for meta in results.get("metadatas", [[]])[0]:
        if meta:  # guard against any None metadata entry
            sections.add(meta.get("section", "unknown"))
    return sorted(sections)


def section_accuracy(retrieved_sections: list, expected_section: str) -> bool:
    """True if expected_section appears in the retrieved sections."""
    if not expected_section:
        return True
    return expected_section in retrieved_sections


def load_questions(csv_path: Path) -> list:
    questions = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            year_val = row.get("year", "").strip()
            row["year"] = int(year_val) if year_val.isdigit() else None
            questions.append(row)
    print(f"Loaded {len(questions)} questions from {csv_path}")
    return questions


def load_existing_success(path: Path) -> dict:
    """
    For --resume: return {(pipeline, question): row_dict} for every row in an
    existing results.csv that previously SUCCEEDED (no error). Errored or
    missing rows are intentionally excluded so they get re-run.
    """
    done = {}
    if not path.exists():
        return done
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not (r.get("error") or "").strip():
                done[(r["pipeline"], r["question"])] = r
    return done


# ============================================================
# RATE-LIMIT-AWARE PIPELINE CALL
# ============================================================

def _is_rate_limit_error(e: Exception) -> bool:
    """Detect Groq/HTTP rate-limit errors (string-based, dependency-light)."""
    s = str(e).lower()
    name = type(e).__name__.lower()
    return (
        "429" in s
        or "rate limit" in s
        or "rate_limit" in s
        or "too many requests" in s
        or "ratelimit" in name
    )


def _run_pipeline_with_backoff(pipeline, question, company, year,
                               derived_section, max_retries, base_backoff):
    """
    Call the requested pipeline, retrying ONLY on rate-limit errors with
    exponential backoff. Any non-rate-limit exception is re-raised so the
    caller's existing error handling records it as an error row.
    """
    attempt = 0
    while True:
        try:
            if pipeline == "baseline":
                return baseline_answer(
                    question=question, company=company,
                    year=year, section=derived_section,
                )
            return agentic_answer(
                question=question, company=company, year=year,
            )
        except Exception as e:
            if _is_rate_limit_error(e) and attempt < max_retries:
                wait = base_backoff * (2 ** attempt)
                print(f"  Rate limit hit; backing off {wait:.1f}s "
                      f"(attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait)
                attempt += 1
                continue
            raise


# ============================================================
# RUN SINGLE QUESTION — one pipeline
# ============================================================

def run_one(
    pipeline: str,
    question: str,
    company: str,
    year,
    expected_section: str,
    gold_answer: str,
    answer_type: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_backoff: float = DEFAULT_BASE_BACKOFF,
) -> dict:
    """Run one question through one pipeline; return a fully-populated row."""

    start = time.time()

    try:

        if pipeline == "baseline":
            # For baseline, derive section from query type for fair comparison
            strategy = build_retrieval_strategy(question)
            derived_section = strategy.get("section")
        else:
            derived_section = None

        response = _run_pipeline_with_backoff(
            pipeline=pipeline,
            question=question,
            company=company if company else None,
            year=year,
            derived_section=derived_section,
            max_retries=max_retries,
            base_backoff=base_backoff,
        )

        elapsed = round(time.time() - start, 2)

        answer = response.get("answer", "")
        results = response.get("results", {"ids": [[]], "metadatas": [[]], "distances": [[]]})
        retry_used = response.get("retry_used", False)
        avg_similarity = response.get("avg_similarity", 0.0)
        top_similarity = response.get("top_similarity", 0.0)
        retrieved_chunks = response.get("retrieved_chunks", 0)

        retrieved_sections = extract_sections_from_results(results)
        sec_acc = section_accuracy(retrieved_sections, expected_section)
        has_citations = citations_present(answer)
        n_citations = count_citations(answer)

        # CRASH FIX: `or {}` handles BOTH a missing key AND an explicit None
        # (the 0-chunk guard returns validation=None).
        validation = response.get("validation") or {}
        years_coverage_ok = validation.get("years_coverage_ok", True)
        numeric_present = validation.get("numeric_present", True)
        overall_valid = validation.get("overall_valid", None)

        return {
            "pipeline": pipeline,
            "question": question,
            "company": company,
            "year": year,
            "answer_type": answer_type,
            "expected_section": expected_section,
            "gold_answer": gold_answer,
            "generated_answer": answer,
            "response_time_sec": elapsed,
            "retry_used": retry_used,
            "avg_similarity": avg_similarity,
            "top_similarity": top_similarity,
            "retrieved_chunks": retrieved_chunks,
            "retrieved_sections": "|".join(retrieved_sections),
            "section_accuracy": sec_acc,
            "citations_present": has_citations,
            "citation_count": n_citations,
            "years_coverage_ok": years_coverage_ok,
            "numeric_present": numeric_present,
            "overall_valid": overall_valid,
            "error": "",
        }

    except Exception as e:
        elapsed = round(time.time() - start, 2)
        print(f"  ERROR on '{pipeline}' for: {question[:60]}")
        traceback.print_exc()
        return {
            "pipeline": pipeline,
            "question": question,
            "company": company,
            "year": year,
            "answer_type": answer_type,
            "expected_section": expected_section,
            "gold_answer": gold_answer,
            "generated_answer": "",
            "response_time_sec": elapsed,
            "retry_used": False,
            "avg_similarity": 0.0,
            "top_similarity": 0.0,
            "retrieved_chunks": 0,
            "retrieved_sections": "",
            "section_accuracy": False,
            "citations_present": False,
            "citation_count": 0,
            "years_coverage_ok": False,
            "numeric_present": False,
            "overall_valid": False,
            "error": str(e),
        }


# ============================================================
# SUMMARY STATS — print at end so you can see improvements
# ============================================================

def print_summary(rows: list):
    """Print aggregate metrics per pipeline for quick comparison."""

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    n_err = sum(1 for r in rows if (r.get("error") or "").strip())
    if n_err:
        print(f"  NOTE: {n_err}/{len(rows)} rows still have errors "
              f"(see 'error' column). Re-run with --resume to retry them.")

    for pipeline in ["baseline", "agentic"]:
        pipe_rows = [r for r in rows if r["pipeline"] == pipeline]
        if not pipe_rows:
            continue

        # Only average over rows that actually produced an answer.
        ok_rows = [r for r in pipe_rows if not (r.get("error") or "").strip()]
        if not ok_rows:
            print(f"\n--- {pipeline.upper()} --- (no successful rows)")
            continue

        def fnum(r, k):
            try:
                return float(r[k])
            except (TypeError, ValueError):
                return 0.0

        def fbool(r, k):
            return str(r.get(k)).strip().lower() == "true"

        n = len(ok_rows)
        avg_latency = sum(fnum(r, "response_time_sec") for r in ok_rows) / n
        avg_sim = sum(fnum(r, "avg_similarity") for r in ok_rows) / n
        top_sim = sum(fnum(r, "top_similarity") for r in ok_rows) / n
        sec_acc_pct = sum(1 for r in ok_rows if fbool(r, "section_accuracy")) / n * 100
        cit_pct = sum(1 for r in ok_rows if fbool(r, "citations_present")) / n * 100
        retry_pct = sum(1 for r in ok_rows if fbool(r, "retry_used")) / n * 100

        print(f"\n--- {pipeline.upper()} ---  (averaged over {n} successful rows)")
        print(f"  Avg Latency:        {avg_latency:.2f} s")
        print(f"  Avg Similarity:     {avg_sim:.2f} %")
        print(f"  Top Similarity:     {top_sim:.2f} %")
        print(f"  Section Accuracy:   {sec_acc_pct:.1f} %")
        print(f"  Citation Rate:      {cit_pct:.1f} %")
        print(f"  Retry Frequency:    {retry_pct:.1f} %")

        # Per-type section accuracy
        type_breakdown = {}
        for r in ok_rows:
            t = r["answer_type"]
            d = type_breakdown.setdefault(t, {"n": 0, "sec_acc": 0, "cit": 0})
            d["n"] += 1
            if fbool(r, "section_accuracy"):
                d["sec_acc"] += 1
            if fbool(r, "citations_present"):
                d["cit"] += 1
        print(f"\n  Per-Type Section Accuracy (successful rows):")
        for t in sorted(type_breakdown):
            s = type_breakdown[t]
            print(f"    {t:14s} n={s['n']:2d}  sec_acc={s['sec_acc']/s['n']*100:5.1f}%  "
                  f"cit={s['cit']/s['n']*100:5.1f}%")

    print("\n" + "=" * 70)


# ============================================================
# NUMERIC GRADER (auto-chained — free, no API)
# ============================================================

def run_numeric_grader():
    print("\n" + "#" * 70)
    print("# STEP 2/2 — NUMERIC GRADER (objective figure matching, no API)")
    print("#" * 70)
    try:
        from evaluation.numeric_grader import grade
        grade()
    except SystemExit:
        print("  Numeric grader exited early (results.csv missing?).")
    except Exception as e:
        print(f"  Numeric grading skipped due to error: {e}")


# ============================================================
# MAIN EVALUATION LOOP
# ============================================================

def run_evaluation(
    sleep_between: float = DEFAULT_SLEEP,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_backoff: float = DEFAULT_BASE_BACKOFF,
    limit: int = None,
    do_grade: bool = True,
    resume: bool = False,
):

    questions = load_questions(QUESTIONS_CSV)
    if limit is not None:
        questions = questions[:limit]
        print(f"Limiting run to the first {len(questions)} question(s).")

    existing = load_existing_success(RESULTS_CSV) if resume else {}
    if resume:
        print(f"RESUME: {len(existing)} previously-successful rows will be "
              f"kept; only missing/errored rows will be re-run.")

    all_rows = []
    total = len(questions)
    completed = 0
    reused = 0
    reran = 0

    print("\n" + "#" * 70)
    print("# STEP 1/2 — GENERATING ANSWERS (baseline + agentic, both 70B)")
    print(f"# sleep={sleep_between}s/call  |  429 backoff up to {max_retries} retries"
          f"{'  |  RESUME' if resume else ''}")
    print("#" * 70)

    for i, q in enumerate(questions):

        question = q["question"]
        company = q.get("company", "").strip() or None
        year = q["year"]
        expected_section = q.get("expected_section", "").strip()
        gold_answer = q.get("gold_answer", "").strip()
        answer_type = q.get("answer_type", "").strip()

        print(f"\n{'='*70}")
        print(f"[{i+1}/{total}] {question}")
        print(f"  Company: {company} | Year: {year} | Type: {answer_type}")

        for pipeline in ["baseline", "agentic"]:
            key = (pipeline, question)

            if resume and key in existing:
                all_rows.append(existing[key])
                reused += 1
                print(f"  --- {pipeline.upper()}: reusing previous success ---")
                continue

            print(f"\n  --- Running: {pipeline.upper()} ---")
            row = run_one(
                pipeline=pipeline, question=question, company=company,
                year=year, expected_section=expected_section,
                gold_answer=gold_answer, answer_type=answer_type,
                max_retries=max_retries, base_backoff=base_backoff,
            )
            all_rows.append(row)
            reran += 1

            if sleep_between > 0:
                time.sleep(sleep_between)

        completed += 1
        print(f"\n  [Progress: {completed}/{total} questions]")

    # --------------------------------------------------------
    # Write results CSV
    # --------------------------------------------------------

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n{'='*70}")
    print(f"Saved {len(all_rows)} rows to: {RESULTS_CSV}")
    if resume:
        print(f"  reused {reused} previous rows, re-ran {reran}.")
    still_err = sum(1 for r in all_rows if (r.get("error") or "").strip())
    print(f"  rows still erroring: {still_err}")

    print_summary(all_rows)

    if do_grade:
        run_numeric_grader()

    print(f"\n{'='*70}")
    print("DONE (generation + numeric grade).")
    if still_err:
        print(f"  {still_err} rows still failed (likely 429). On a fresh-quota "
              f"day run:  python -m evaluation.run_evaluation --resume --sleep 6")
    print("  View now:   streamlit run evaluation/dashboard.py")
    print("  Add judge:  python -m evaluation.llm_judge   (separate, 70B)")
    print(f"{'='*70}")

    return all_rows


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run the RAG evaluation (baseline + agentic) and the free "
                    "numeric grader. LLM judge is separate (evaluation.llm_judge)."
    )
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP,
                        help=f"Pause after each pipeline call (default {DEFAULT_SLEEP}s).")
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES,
                        help=f"Max backoff retries on a 429 (default {DEFAULT_MAX_RETRIES}).")
    parser.add_argument("--base-backoff", type=float, default=DEFAULT_BASE_BACKOFF,
                        help=f"First backoff wait, doubles each retry (default {DEFAULT_BASE_BACKOFF}s).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only run the first N questions (smoke testing).")
    parser.add_argument("--no-grade", action="store_true",
                        help="Skip the numeric grader step (answers only).")
    parser.add_argument("--resume", action="store_true",
                        help="Keep previously-successful rows in results.csv and "
                             "re-run only missing/errored rows.")
    args = parser.parse_args()

    run_evaluation(
        sleep_between=args.sleep,
        max_retries=args.max_retries,
        base_backoff=args.base_backoff,
        limit=args.limit,
        do_grade=not args.no_grade,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()