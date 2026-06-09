"""
src/agent/router.py

Builds dynamic retrieval strategies based on query type.

Updates in this version (additive, nothing removed):
- NEW "properties" query_type -> Properties (Item 2) section, with a
  Business fallback so a thin Properties section never returns empty.
- Cybersecurity risk queries now target the dedicated Cybersecurity
  (Item 1C) section, with a Risk Factors fallback. Other risk queries
  keep the tight Risk Factors filter exactly as before.
- Descriptive queries still allow Business OR MD&A jointly.
- Numeric routing (Financial Statements) is unchanged; the agentic
  pipeline now boosts table chunks within that section at rerank time.
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
    q = question.lower()

    strategy = {
        "query_type": query_type,
        "top_k": 3,
        "section": None,
        "sections_allowed": None,
        "use_comparison": False
    }

    # ========================================================
    # RISK QUESTIONS — Risk Factors, with a Cybersecurity split
    # ========================================================

    if query_type == "risk":
        if "cyber" in q:
            # Cybersecurity disclosures live in Item 1C, but some companies
            # still discuss cyber risk under Risk Factors — allow both.
            strategy["section"] = None
            strategy["sections_allowed"] = ["Cybersecurity", "Risk Factors"]
            strategy["top_k"] = 4
        else:
            strategy["section"] = "Risk Factors"
            strategy["top_k"] = 4

    # ========================================================
    # NUMERIC QUESTIONS — Financial Statements (tables boosted at rerank)
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
    # PROPERTIES QUESTIONS (NEW) — Item 2, with Business fallback
    # ========================================================

    elif query_type == "properties":
        strategy["section"] = None
        strategy["sections_allowed"] = ["Properties", "Business"]
        strategy["top_k"] = 4

    # ========================================================
    # DESCRIPTIVE QUESTIONS — Allow Business OR MD&A jointly
    # ========================================================

    elif query_type == "descriptive":
        strategy["section"] = None
        strategy["sections_allowed"] = ["Business", "MD&A"]
        strategy["top_k"] = 5

    # ========================================================
    # COMPARATIVE QUESTIONS — Section depends on sub-type
    # ========================================================

    elif query_type == "comparative":

        strategy["use_comparison"] = True
        strategy["top_k"] = 6

        # Comparative cybersecurity
        if "cyber" in q:
            strategy["sections_allowed"] = ["Cybersecurity", "Risk Factors"]

        # Comparative risk queries
        elif any(word in q for word in ["risk", "risks", "challenge", "threat"]):
            strategy["section"] = "Risk Factors"

        # Comparative financial queries
        elif any(word in q for word in ["revenue", "income", "profit", "sales", "earnings"]):
            strategy["section"] = "Financial Statements"

        # Comparative properties queries
        elif any(word in q for word in ["properties", "property", "facilities", "headquarters"]):
            strategy["sections_allowed"] = ["Properties", "Business"]

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
        "Compare Apple's business strategy across 2023 and 2024",
        "What properties does Apple own or lease?",
        "Describe Microsoft's cybersecurity risk governance",
    ]

    for q in questions:
        strategy = build_retrieval_strategy(q)
        print("\n" + "=" * 50)
        print(f"Question: {q}")
        print(f"Strategy: {strategy}")