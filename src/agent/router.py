"""
src/agent/router.py

Builds dynamic retrieval strategies based on query type.

Updates:
- Descriptive queries now allow Business OR MD&A jointly (via sections_allowed)
- All other behaviors preserved from previous version
"""

from src.agent.query_classifier import classify_query


def build_retrieval_strategy(question):
    """
    Build retrieval strategy dynamically based on query type.

    Returns a dict with:
    - query_type: classified type
    - top_k: number of chunks to retrieve
    - section: single section filter (or None)
    - sections_allowed: list of sections to OR-filter (or None)
    - use_comparison: flag for comparative queries
    """

    query_type = classify_query(question)

    strategy = {
        "query_type": query_type,
        "top_k": 3,
        "section": None,
        "sections_allowed": None,
        "use_comparison": False
    }

    # ========================================================
    # RISK QUESTIONS — Tight section filter
    # ========================================================

    if query_type == "risk":
        strategy["section"] = "Risk Factors"
        strategy["top_k"] = 4

    # ========================================================
    # NUMERIC QUESTIONS — Tight section filter
    # ========================================================

    elif query_type == "numeric":
        strategy["section"] = "Financial Statements"
        strategy["top_k"] = 5

    # ========================================================
    # SUMMARY QUESTIONS — MD&A
    # ========================================================

    elif query_type == "summary":
        strategy["section"] = "MD&A"
        strategy["top_k"] = 4

    # ========================================================
    # DESCRIPTIVE QUESTIONS — Allow Business OR MD&A jointly
    # ========================================================

    elif query_type == "descriptive":
        # Allow embedding model to pick whichever section is most relevant
        strategy["section"] = None
        strategy["sections_allowed"] = ["Business", "MD&A"]
        strategy["top_k"] = 5

    # ========================================================
    # COMPARATIVE QUESTIONS — Section depends on sub-type
    # ========================================================

    elif query_type == "comparative":

        strategy["use_comparison"] = True
        strategy["top_k"] = 6

        q = question.lower()

        # Comparative risk queries
        if any(word in q for word in ["risk", "risks", "challenge", "threat"]):
            strategy["section"] = "Risk Factors"

        # Comparative financial queries
        elif any(word in q for word in ["revenue", "income", "profit", "sales", "earnings"]):
            strategy["section"] = "Financial Statements"

        # Comparative summary/outlook queries
        elif any(word in q for word in ["outlook", "summary", "overview"]):
            strategy["section"] = "MD&A"

        # Comparative descriptive — allow Business + MD&A
        else:
            strategy["sections_allowed"] = ["Business", "MD&A"]

    return strategy


# ============================================================
# TESTS
# ============================================================

if __name__ == "__main__":

    questions = [
        "What were Apple's major risk factors?",
        "What was Tesla's revenue in 2024?",
        "Compare Microsoft's risks between 2023 and 2024",
        "Summarize Amazon's outlook",
        "Describe Google's cloud business strategy",
        "Compare Apple's business strategy across 2023 and 2024"
    ]

    for q in questions:
        strategy = build_retrieval_strategy(q)
        print("\n" + "=" * 50)
        print(f"Question: {q}")
        print(f"Strategy: {strategy}")