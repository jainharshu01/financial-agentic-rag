"""
evaluation/run_evaluation.py

Automated evaluation pipeline.

- Loads evaluation_questions.csv
- Runs EVERY question through baseline_rag and agentic_rag
- Stores results to evaluation/results.csv
- No manual question input required.

Run:
    python -m evaluation.run_evaluation
    OR
    python evaluation/run_evaluation.py
"""

import os
import sys
import csv
import time
import re
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
# HELPERS
# ============================================================

CITATION_PATTERN = re.compile(r"\(source\s*\d+\)", re.IGNORECASE)


def citations_present(answer: str) -> bool:
    return bool(CITATION_PATTERN.search(answer))


def extract_sections_from_results(results: dict) -> list:
    """Return list of unique section names from retrieved chunks."""
    sections = set()
    for meta in results.get("metadatas", [[]])[0]:
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
            # Convert year to int if present
            year_val = row.get("year", "").strip()
            row["year"] = int(year_val) if year_val.isdigit() else None
            questions.append(row)
    print(f"Loaded {len(questions)} questions from {csv_path}")
    return questions


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
) -> dict:
    """
    Run a single question through the specified pipeline.

    Args:
        pipeline: 'baseline' or 'agentic'

    Returns:
        dict with all logged fields.
    """

    start = time.time()

    try:

        if pipeline == "baseline":
            # For baseline, derive section from query type for fair comparison
            strategy = build_retrieval_strategy(question)
            derived_section = strategy.get("section")

            response = baseline_answer(
                question=question,
                company=company if company else None,
                year=year,
                section=derived_section,
            )
        else:
            response = agentic_answer(
                question=question,
                company=company if company else None,
                year=year,
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

        # Validation dict (agentic only)
        validation = response.get("validation", {})
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
            "years_coverage_ok": False,
            "numeric_present": False,
            "overall_valid": False,
            "error": str(e),
        }


# ============================================================
# MAIN EVALUATION LOOP
# ============================================================

def run_evaluation():

    questions = load_questions(QUESTIONS_CSV)

    all_rows = []

    total = len(questions)
    completed = 0

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
            print(f"\n  --- Running: {pipeline.upper()} ---")
            row = run_one(
                pipeline=pipeline,
                question=question,
                company=company,
                year=year,
                expected_section=expected_section,
                gold_answer=gold_answer,
                answer_type=answer_type,
            )
            all_rows.append(row)

        completed += 1
        print(f"\n  [Progress: {completed}/{total} questions complete]")

    # --------------------------------------------------------
    # Write results CSV
    # --------------------------------------------------------

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "pipeline", "question", "company", "year", "answer_type",
        "expected_section", "gold_answer", "generated_answer",
        "response_time_sec", "retry_used", "avg_similarity", "top_similarity",
        "retrieved_chunks", "retrieved_sections", "section_accuracy",
        "citations_present", "years_coverage_ok", "numeric_present",
        "overall_valid", "error",
    ]

    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n{'='*70}")
    print(f"Evaluation complete. Results saved to: {RESULTS_CSV}")
    print(f"Total rows: {len(all_rows)} ({total} questions × 2 pipelines)")

    return all_rows


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_evaluation()