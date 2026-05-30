"""
evaluation/llm_judge.py   (NEW)

LLM-as-judge for prose answers (risk + descriptive questions).

WHY
---
Numeric questions are graded objectively by numeric_grader.py. But the
18 risk/descriptive questions have prose gold answers like:
    "Apple faces risks including global competition, supply chain
    concentration in China, intellectual property litigation, ..."
There's no single number to match. We need a model to judge whether the
generated answer covers the gold's key points.

DESIGN
------
- Uses your existing Groq client (no new API key, no new dependency).
- Asks the LLM to score 0.0-1.0 on coverage of the gold's key points,
  plus a one-line justification.
- Skips numeric/comparative rows (those go through numeric_grader.py).
- Reads results.csv, writes results_judged.csv with two new columns:
  judge_score and judge_reason.
- Polite: 0.7s sleep between calls so we don't hammer the API.

USAGE
-----
    python -m evaluation.llm_judge

WHEN TO TRUST IT
----------------
The judge is a model, not ground truth. Treat scores >= 0.7 as "covers
the key points", < 0.4 as "misses". Always spot-check 3-5 rows yourself
the first time, to verify the judge matches your intuition.
"""

import csv
import os
import sys
import time
import re
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_CSV = EVAL_DIR / "results.csv"
OUT_CSV = EVAL_DIR / "results_judged.csv"

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"
SLEEP_BETWEEN_CALLS = 0.7  # seconds


JUDGE_PROMPT = """You are an impartial evaluator scoring how well a
generated answer covers the key points of a reference (gold) answer.

QUESTION:
{question}

GOLD ANSWER (reference):
{gold}

GENERATED ANSWER:
{generated}

Score the generated answer on a scale from 0.0 to 1.0 based ONLY on
whether it covers the key facts and points present in the gold answer.

Scoring rubric:
  1.0  = covers all key points; equivalent or better
  0.8  = covers most key points; minor gaps
  0.6  = covers some key points; meaningful gaps
  0.4  = covers a few key points; many missing
  0.2  = barely related
  0.0  = unrelated, refuses, or says "insufficient evidence"

Respond in EXACTLY this format on two lines:
SCORE: <number between 0.0 and 1.0>
REASON: <one short sentence>
"""


def parse_response(text: str):
    """Pull score and reason out of the model's response."""
    score = None
    reason = ""
    m = re.search(r"SCORE\s*:\s*([01](?:\.\d+)?)", text, re.IGNORECASE)
    if m:
        try:
            score = float(m.group(1))
            score = max(0.0, min(1.0, score))
        except ValueError:
            score = None
    m = re.search(r"REASON\s*:\s*(.+)", text, re.IGNORECASE)
    if m:
        reason = m.group(1).strip()
    return score, reason


def judge_one(question, gold, generated):
    prompt = JUDGE_PROMPT.format(
        question=question, gold=gold, generated=generated
    )
    try:
        resp = groq_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,  # deterministic judging
        )
        text = resp.choices[0].message.content
        return parse_response(text)
    except Exception as e:
        return None, f"ERROR: {e}"


def run():
    if not RESULTS_CSV.exists():
        print(f"ERROR: {RESULTS_CSV} not found. Run run_evaluation first.")
        sys.exit(1)

    rows = list(csv.DictReader(open(RESULTS_CSV, encoding="utf-8")))
    fieldnames = list(rows[0].keys()) + ["judge_score", "judge_reason"]

    total = sum(
        1 for r in rows if r["answer_type"] in ("risk", "descriptive")
    )
    print(f"Judging {total} prose answers (risk + descriptive) ...\n")

    judged = []
    done = 0
    for r in rows:
        if r["answer_type"] in ("risk", "descriptive"):
            done += 1
            print(f"[{done}/{total}] {r['pipeline']:8s} | "
                  f"{r['question'][:55]} ...")
            score, reason = judge_one(
                r["question"], r.get("gold_answer", ""),
                r.get("generated_answer", ""),
            )
            r["judge_score"] = "" if score is None else f"{score:.2f}"
            r["judge_reason"] = reason
            print(f"    -> score={r['judge_score']}  {reason[:80]}")
            time.sleep(SLEEP_BETWEEN_CALLS)
        else:
            # Numeric / comparative: leave blank, graded by numeric_grader
            r["judge_score"] = ""
            r["judge_reason"] = ""
        judged.append(r)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(judged)

    print(f"\nWrote {OUT_CSV}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("LLM-AS-JUDGE SUMMARY (prose answers only)")
    print("=" * 60)
    stats = {}
    for r in judged:
        if not r["judge_score"]:
            continue
        p = r["pipeline"]
        stats.setdefault(p, []).append(float(r["judge_score"]))

    for p, scores in stats.items():
        avg = sum(scores) / len(scores)
        good = sum(1 for s in scores if s >= 0.7)
        weak = sum(1 for s in scores if s < 0.4)
        print(f"  {p:9s}: n={len(scores):2d}  avg={avg:.2f}  "
              f">=0.7: {good}  <0.4: {weak}")


if __name__ == "__main__":
    run()