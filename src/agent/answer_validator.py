"""
src/agent/answer_validator.py

Validates generated answers for:
1. Citation presence — does the answer cite (Source X) ?
2. Unsupported claims — does the answer contain hedge phrases implying
   no evidence was found despite chunks being available?
3. Both-year coverage — for comparative queries, are both years mentioned?
4. Numeric consistency — if a numeric answer is expected, is a number present?

Returns a structured validation dict so the agentic pipeline can decide
whether to flag or re-generate.
"""

import re
from src.agent.query_classifier import classify_query
from src.agent.query_parser import extract_years


# ============================================================
# CONSTANTS
# ============================================================

# Phrases that strongly suggest the model gave up / hallucinated absence
UNSUPPORTED_PHRASES = [
    "no relevant information",
    "unable to answer",
    "documents do not contain",
    "context does not contain",
    "cannot be determined",
    "not mentioned in",
    "not provided in",
    "no information available",
    "not stated in the",
    "not found in the",
    "insufficient evidence",
    "no data available",
]

CITATION_PATTERN = re.compile(r"\(source\s*\d+\)", re.IGNORECASE)
NUMBER_PATTERN = re.compile(r"\b\d[\d,\.]*\s*(?:billion|million|thousand|%|percent)?\b", re.IGNORECASE)


# ============================================================
# VALIDATOR
# ============================================================

def validate_answer(
    question: str,
    answer: str,
    results: dict
) -> dict:
    """
    Validate a generated answer against several quality signals.

    Args:
        question: Original user question.
        answer:   Generated answer string.
        results:  ChromaDB results dict (to check # of retrieved chunks).

    Returns:
        dict with keys:
            - citations_present (bool)
            - unsupported_claim_detected (bool)
            - unsupported_phrases_found (list[str])
            - years_coverage_ok (bool)   — True if not comparative, or both years present
            - numeric_present (bool)     — True if not numeric query, or number found
            - chunk_count (int)
            - overall_valid (bool)       — True only when all critical checks pass
            - warnings (list[str])
    """

    answer_lower = answer.lower()
    warnings = []

    # --------------------------------------------------------
    # 1. Citation presence
    # --------------------------------------------------------
    citations_present = bool(CITATION_PATTERN.search(answer))
    if not citations_present:
        warnings.append("No citations found in answer (expected '(Source X)' format).")

    # --------------------------------------------------------
    # 2. Unsupported claims
    # --------------------------------------------------------
    found_unsupported = [
        phrase for phrase in UNSUPPORTED_PHRASES
        if phrase in answer_lower
    ]
    unsupported_claim_detected = len(found_unsupported) > 0

    chunk_count = len(results.get("ids", [[]])[0])

    if unsupported_claim_detected and chunk_count >= 3:
        warnings.append(
            f"Model claims insufficient evidence but {chunk_count} chunks were retrieved. "
            f"Phrases detected: {found_unsupported}"
        )

    # --------------------------------------------------------
    # 3. Both-year coverage (comparative queries only)
    # --------------------------------------------------------
    query_type = classify_query(question)
    query_years = extract_years(question)
    years_coverage_ok = True

    if query_type == "comparative" and len(query_years) >= 2:
        missing_years = [str(y) for y in query_years if str(y) not in answer]
        if missing_years:
            years_coverage_ok = False
            warnings.append(
                f"Comparative query detected but year(s) {missing_years} "
                "not mentioned in the answer."
            )

    # --------------------------------------------------------
    # 4. Numeric presence (numeric queries only)
    # --------------------------------------------------------
    numeric_present = True

    if query_type == "numeric":
        numeric_present = bool(NUMBER_PATTERN.search(answer))
        if not numeric_present:
            warnings.append(
                "Numeric query detected but no numbers found in the answer."
            )

    # --------------------------------------------------------
    # Overall validity
    # --------------------------------------------------------
    # Critical failures: unsupported claim with chunks present,
    # or missing year coverage in comparative, or no citations.
    overall_valid = (
        citations_present
        and not (unsupported_claim_detected and chunk_count >= 3)
        and years_coverage_ok
        and numeric_present
    )

    return {
        "citations_present": citations_present,
        "unsupported_claim_detected": unsupported_claim_detected,
        "unsupported_phrases_found": found_unsupported,
        "years_coverage_ok": years_coverage_ok,
        "numeric_present": numeric_present,
        "chunk_count": chunk_count,
        "overall_valid": overall_valid,
        "warnings": warnings,
    }


# ============================================================
# TESTS
# ============================================================

if __name__ == "__main__":

    fake_results = {"ids": [["id1", "id2", "id3"]]}

    tests = [
        {
            "question": "Compare Apple risks between 2023 and 2024",
            "answer": (
                "In 2023, Apple faced supply chain risks (Source 1). "
                "In 2024, cybersecurity threats increased significantly (Source 2)."
            ),
        },
        {
            "question": "What was Tesla's revenue in 2024?",
            "answer": "Insufficient evidence in the provided documents.",
        },
        {
            "question": "What are Apple's risk factors?",
            "answer": "Apple faces competition and supply chain risks.",
        },
    ]

    for t in tests:
        print("\n" + "="*60)
        print(f"Q: {t['question']}")
        print(f"A: {t['answer']}")
        result = validate_answer(t["question"], t["answer"], fake_results)
        for k, v in result.items():
            print(f"  {k}: {v}")