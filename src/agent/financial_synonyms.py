"""
src/agent/financial_synonyms.py

Static financial-term alias map.

Design rationale (research-backed):
- A 2026 text-and-table financial RAG benchmark found that heavy query
  expansion (HyDE, multi-query) gives LIMITED benefit for precise numeric
  queries, while exact lexical matching (BM25) is what actually helps.
- Therefore these aliases are applied ONLY on the BM25 / lexical side of
  retrieval (see bm25_index.py), NOT to the dense embedding query.
- We also use them to normalize evaluation so a correct answer that says
  "net sales" is not penalized against a gold answer that says "revenue".

Keep this map small, high-precision, and SEC-accurate. Apple, for example,
reports "net sales" rather than "revenue".
"""

# Canonical user term -> set of SEC/GAAP-equivalent surface forms.
# All lowercase; matching is case-insensitive downstream.
FINANCIAL_SYNONYMS = {
    "revenue": [
        "net sales", "total revenue", "total revenues", "revenues",
        "total net sales", "net revenue", "net revenues",
    ],
    "profit": [
        "net income", "net earnings", "net profit", "profit",
        "net income loss",
    ],
    "earnings": [
        "diluted eps", "earnings per share", "basic eps",
        "diluted earnings per share", "net income", "eps",
    ],
    "operating profit": [
        "operating income", "income from operations",
        "operating income loss",
    ],
    "operating income": [
        "operating profit", "income from operations",
    ],
    "gross profit": [
        "gross margin", "gross income",
    ],
    "cash flow": [
        "cash flows", "net cash", "cash and cash equivalents",
        "operating cash flow", "cash from operations",
        "net cash provided by operating activities",
    ],
    "assets": [
        "total assets", "current assets", "total current assets",
    ],
    "liabilities": [
        "total liabilities", "current liabilities",
        "total current liabilities",
    ],
    "debt": [
        "long-term debt", "long term debt", "total debt",
        "notes payable",
    ],
    "r&d": [
        "research and development", "research development",
        "r and d",
    ],
    "capex": [
        "capital expenditures", "capital expenditure",
        "purchases of property and equipment",
    ],
}


def expand_query_terms(question: str) -> list:
    """
    Return a list of extra keyword terms to append to the BM25 query.

    For every canonical term found in the question, we add all of its
    SEC-equivalent surface forms. We do NOT touch the dense query.

    Args:
        question: the user's natural-language question.

    Returns:
        A de-duplicated list of additional lexical terms (may be empty).
    """
    if not question:
        return []

    q = question.lower()
    extra = []
    seen = set()

    for canonical, aliases in FINANCIAL_SYNONYMS.items():
        # Match either the canonical key or any alias already present.
        trigger = canonical in q or any(a in q for a in aliases)
        if not trigger:
            continue
        for term in [canonical] + aliases:
            if term not in seen and term not in q:
                extra.append(term)
                seen.add(term)

    return extra


def normalize_for_eval(text: str) -> str:
    """
    Collapse known synonyms to a canonical token so evaluation overlap
    checks don't penalize valid SEC terminology.

    Replaces every alias with its canonical key (longest aliases first to
    avoid partial clobbering).
    """
    if not text:
        return ""

    out = text.lower()

    # Build (alias, canonical) pairs sorted by alias length desc.
    pairs = []
    for canonical, aliases in FINANCIAL_SYNONYMS.items():
        for a in aliases:
            pairs.append((a, canonical))
    pairs.sort(key=lambda p: len(p[0]), reverse=True)

    for alias, canonical in pairs:
        out = out.replace(alias, canonical)

    return out


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    tests = [
        "What was Apple's revenue in 2024?",
        "Compare Tesla's net income across 2023 and 2024",
        "What were Microsoft's diluted EPS?",
        "Describe Google's business",  # no financial term
    ]
    for t in tests:
        print(f"\nQ: {t}")
        print(f"  expand -> {expand_query_terms(t)}")
        print(f"  norm   -> {normalize_for_eval(t)}")