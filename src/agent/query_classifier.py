import re


def classify_query(question):
    """
    Rule-based query classifier.
    """

    q = question.lower()

    # ========================================================
    # COMPARATIVE QUESTIONS
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
    # RISK QUESTIONS
    # ========================================================

    risk_keywords = [
        "risk",
        "risks",
        "uncertainty",
        "threat",
        "challenge",
        "cybersecurity"
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

        "What challenges did Google mention?"
    ]

    for q in test_questions:

        result = classify_query(q)

        print(f"\nQuestion: {q}")
        print(f"Classified as: {result}")