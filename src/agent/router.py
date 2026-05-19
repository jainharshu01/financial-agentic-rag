from query_classifier import classify_query


def build_retrieval_strategy(question):
    """
    Build retrieval strategy dynamically
    based on query type.
    """

    query_type = classify_query(question)

    strategy = {
        "query_type": query_type,
        "top_k": 3,
        "section": None,
        "use_comparison": False
    }

    # ========================================================
    # RISK QUESTIONS
    # ========================================================

    if query_type == "risk":

        strategy["section"] = "Risk Factors"
        strategy["top_k"] = 4

    # ========================================================
    # NUMERIC QUESTIONS
    # ========================================================

    elif query_type == "numeric":

        strategy["section"] = "Financial Statements"
        strategy["top_k"] = 5

    # ========================================================
    # SUMMARY QUESTIONS
    # ========================================================

    elif query_type == "summary":

        strategy["section"] = "MD&A"
        strategy["top_k"] = 4

    # ========================================================
    # COMPARATIVE QUESTIONS
    # ========================================================

    elif query_type == "comparative":

        strategy["use_comparison"] = True
        strategy["top_k"] = 6

        q = question.lower()

        # Comparative risk queries
        if any(word in q for word in [
            "risk",
            "risks",
            "challenge",
            "threat"
        ]):

            strategy["section"] = "Risk Factors"

        # Comparative financial queries
        elif any(word in q for word in [
            "revenue",
            "income",
            "profit",
            "sales",
            "earnings"
        ]):

            strategy["section"] = "Financial Statements"

        # Comparative summary/outlook queries
        elif any(word in q for word in [
            "outlook",
            "summary",
            "overview"
        ]):

            strategy["section"] = "MD&A"
    return strategy

# ============================================================
# TESTS
# ============================================================

if __name__ == "__main__":

    questions = [

        "What were Apple's major risk factors?",

        "What was Tesla's revenue in 2024?",

        "Compare Microsoft's risks between 2023 and 2024",

        "Summarize Amazon's outlook"
    ]

    for q in questions:

        strategy = build_retrieval_strategy(q)

        print("\n" + "="*50)
        print(f"Question: {q}")
        print(f"Strategy: {strategy}")