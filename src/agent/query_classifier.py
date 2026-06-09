import re


def classify_query(question):
    """
    Rule-based query classifier.

    CHANGE: added a "properties" type. Previously a question like
    "What properties does Apple own?" matched no keyword and fell through
    to "descriptive", which routes to Business/MD&A — so the Properties
    (Item 2) section was never retrievable. Cybersecurity questions still
    classify as "risk"; the router now sends them to the dedicated
    Cybersecurity (Item 1C) section with a Risk Factors fallback.
    """

    q = question.lower()

    # ========================================================
    # COMPARATIVE QUESTIONS  (checked first)
    # ========================================================

    comparative_keywords = [
        "compare",
        "difference",
        "changed",
        "change",
        "compared",
        "versus",
        "vs"
    ]

    if any(word in q for word in comparative_keywords):
        return "comparative"

    # ========================================================
    # NUMERIC QUESTIONS
    # ========================================================

    numeric_keywords = [
        "revenue",
        "net income",
        "profit",
        "loss",
        "assets",
        "cash flow",
        "sales",
        "operating income",
        "earnings"
    ]

    if any(word in q for word in numeric_keywords):
        return "numeric"

    # ========================================================
    # RISK QUESTIONS  (includes cybersecurity; router splits it)
    # ========================================================

    risk_keywords = [
        "risk",
        "risks",
        "uncertainty",
        "threat",
        "challenge",
        "cybersecurity",
        "cyber"
    ]

    if any(word in q for word in risk_keywords):
        return "risk"

    # ========================================================
    # SUMMARY QUESTIONS
    # ========================================================

    summary_keywords = [
        "summarize",
        "summary",
        "overview",
        "outlook"
    ]

    if any(word in q for word in summary_keywords):
        return "summary"

    # ========================================================
    # PROPERTIES QUESTIONS  (NEW — Item 2)
    # ========================================================

    properties_keywords = [
        "properties",
        "property",
        "real estate",
        "headquarters",
        "office space",
        "manufacturing facilities",
        "leased",
        "owned premises",
        "data centers",
        "data centres",
    ]

    if any(word in q for word in properties_keywords):
        return "properties"

    # ========================================================
    # DEFAULT
    # ========================================================

    return "descriptive"


# ============================================================
# TESTS
# ============================================================

if __name__ == "__main__":

    test_questions = [
        "What were Apple's risk factors?",
        "What was Tesla's revenue in 2024?",
        "Compare Microsoft's risks between 2023 and 2024",
        "Summarize Amazon's business outlook",
        "What challenges did Google mention?",
        "What properties does Apple own or lease?",
        "Describe Microsoft's cybersecurity governance",
    ]

    for q in test_questions:
        result = classify_query(q)
        print(f"\nQuestion: {q}")
        print(f"Classified as: {result}")