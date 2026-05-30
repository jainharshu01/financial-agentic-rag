"""
src/agent/answer_validator.py

Validates generated answers against multiple quality dimensions.

Updates:
- Citation regex now matches (Source N), [Source N], and bare Source N
- All three formats count as valid citations
"""

import re

# ============================================================
# FLEXIBLE CITATION REGEX
# ============================================================
# Matches:
#   (Source 1), (source 2)
#   [Source 3]
#   Source 4, source 5
# Captures the numeric ID for downstream styling.

CITATION_PATTERN = re.compile(
    r"[\(\[]?\s*source\s*(\d+)\s*[\)\]]?",
    re.IGNORECASE
)


def check_citations_present(answer):
    """
    Return True if the answer contains at least one citation
    in any of the supported formats.
    """
    if not answer:
        return False
    return bool(CITATION_PATTERN.search(answer))


def count_citations(answer):
    """Count total citation occurrences in the answer."""
    if not answer:
        return 0
    return len(CITATION_PATTERN.findall(answer))


def get_cited_source_ids(answer):
    """Return the set of source IDs cited in the answer."""
    if not answer:
        return set()
    return set(CITATION_PATTERN.findall(answer))


# ============================================================
# YEAR COVERAGE CHECK (for comparative queries)
# ============================================================

def extract_years_from_text(text):
    """Pull all 4-digit years (20XX) from a string."""
    return set(re.findall(r"\b20\d{2}\b", text))


def check_year_coverage(answer, query_years):
    """
    For comparative queries:
    Return True if all years from the query appear in the answer.
    """
    if not query_years:
        return True
    answer_years = extract_years_from_text(answer)
    return all(str(y) in answer_years for y in query_years)


# ============================================================
# NUMERIC PRESENCE CHECK (for numeric queries)
# ============================================================

def check_numeric_present(answer):
    """
    Return True if the answer contains at least one number.
    Matches integers, decimals, percentages, and dollar amounts.
    """
    if not answer:
        return False
    return bool(re.search(r"\d", answer))


# ============================================================
# UNSUPPORTED CLAIM DETECTION
# ============================================================

UNSUPPORTED_PHRASES = [
    "i don't know",
    "i cannot determine",
    "outside the scope",
    "based on my knowledge",
    "as an ai",
    "i don't have access"
]


def check_unsupported_claim(answer):
    """
    Detect phrases that suggest the model is going outside the provided context.
    Returns True if an unsupported-claim phrase is detected.
    """
    if not answer:
        return False
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in UNSUPPORTED_PHRASES)


# ============================================================
# MAIN VALIDATOR
# ============================================================

def validate_answer(answer, query, query_type, query_years=None):
    """
    Validate an answer across all dimensions.

    Returns a dict:
    {
        "citations_present": bool,
        "citation_count": int,
        "unsupported_claim_detected": bool,
        "years_coverage_ok": bool,
        "numeric_present": bool,
        "overall_valid": bool,
        "warnings": [str, ...]
    }
    """
    warnings = []

    citations_present = check_citations_present(answer)
    citation_count = count_citations(answer)
    unsupported = check_unsupported_claim(answer)

    # Year coverage check (only meaningful for comparative queries)
    if query_type == "comparative" and query_years:
        years_ok = check_year_coverage(answer, query_years)
        if not years_ok:
            missing = [
                str(y) for y in query_years
                if str(y) not in extract_years_from_text(answer)
            ]
            warnings.append(f"Missing year(s) in answer: {', '.join(missing)}")
    else:
        years_ok = True

    # Numeric presence check (only meaningful for numeric queries)
    if query_type == "numeric":
        numeric_ok = check_numeric_present(answer)
        if not numeric_ok:
            warnings.append("Numeric query but no numbers in answer.")
    else:
        numeric_ok = True

    if not citations_present:
        warnings.append("No citations detected in answer.")

    if unsupported:
        warnings.append("Possible unsupported claim detected.")

    # Overall validity = all checks pass
    overall_valid = (
        citations_present
        and not unsupported
        and years_ok
        and numeric_ok
    )

    return {
        "citations_present": citations_present,
        "citation_count": citation_count,
        "unsupported_claim_detected": unsupported,
        "years_coverage_ok": years_ok,
        "numeric_present": numeric_ok,
        "overall_valid": overall_valid,
        "warnings": warnings
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_cases = [
        {
            "answer": "Apple's revenue was $383B in 2024 (Source 1). Cloud grew 15% (Source 2).",
            "query": "What was Apple's revenue?",
            "query_type": "numeric",
            "query_years": [2024]
        },
        {
            "answer": "Microsoft cited cybersecurity threats Source 1 and supply chain risks Source 2.",
            "query": "What are Microsoft's risks?",
            "query_type": "risk",
            "query_years": None
        },
        {
            "answer": "Tesla had strong growth in 2023 [Source 1] and continued expansion in 2024 [Source 2].",
            "query": "Compare Tesla growth between 2023 and 2024",
            "query_type": "comparative",
            "query_years": [2023, 2024]
        },
        {
            "answer": "I don't have access to specific information about this filing.",
            "query": "What was the revenue?",
            "query_type": "numeric",
            "query_years": None
        }
    ]

    for tc in test_cases:
        print("\n" + "=" * 60)
        print(f"Query: {tc['query']}")
        print(f"Answer: {tc['answer']}")
        result = validate_answer(
            tc["answer"], tc["query"], tc["query_type"], tc["query_years"]
        )
        print(f"Validation: {result}")