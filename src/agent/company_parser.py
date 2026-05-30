"""
src/agent/company_parser.py

Extracts company tickers from natural language queries.

Examples:
    "What was Apple's revenue in 2024?" → ["AAPL"]
    "Compare Tesla and Microsoft" → ["TSLA", "MSFT"]
    "Compare Tesla's revenue with Apple's" → ["TSLA", "AAPL"]
"""

import re

# ============================================================
# COMPANY NAME ALIASES → TICKER
# ============================================================

COMPANY_ALIASES = {
    "AAPL": [
        "apple", "apple inc", "aapl"
    ],
    "AMZN": [
        "amazon", "amazon.com", "amzn"
    ],
    "GOOGL": [
        "google", "alphabet", "googl", "goog", "alphabet inc"
    ],
    "MSFT": [
        "microsoft", "microsoft corporation", "msft"
    ],
    "TSLA": [
        "tesla", "tesla inc", "tsla"
    ],
}


def extract_companies(query: str) -> list:
    """
    Extract company tickers mentioned in the query.
    Returns a list of unique tickers in order of first appearance.

    Args:
        query: Natural language question

    Returns:
        List of ticker strings, e.g. ['AAPL', 'TSLA']
    """
    if not query:
        return []

    query_lower = query.lower()
    found = []
    seen = set()

    # Track the position of each match so we return in order of appearance
    positions = []

    for ticker, aliases in COMPANY_ALIASES.items():
        for alias in aliases:
            # Word-boundary regex to avoid 'tesla' matching inside 'teslavolt'
            pattern = rf"\b{re.escape(alias)}\b"
            match = re.search(pattern, query_lower)
            if match and ticker not in seen:
                positions.append((match.start(), ticker))
                seen.add(ticker)
                break  # Only need one alias to match per ticker

    # Sort by position of first match
    positions.sort()
    return [t for _, t in positions]


def is_cross_company_query(query: str) -> bool:
    """Return True if the query mentions 2+ companies."""
    return len(extract_companies(query)) >= 2


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    test_queries = [
        "What was Apple's revenue in 2024?",
        "Compare Tesla and Microsoft",
        "Compare Tesla's revenue with Apple's",
        "How did Amazon and Google perform?",
        "What are the risks for AAPL and TSLA?",
        "Tell me about the tech sector",  # No companies
        "Microsoft vs Apple cloud strategy",
    ]

    for q in test_queries:
        companies = extract_companies(q)
        cross = is_cross_company_query(q)
        print(f"\nQuery: {q}")
        print(f"  Companies: {companies}")
        print(f"  Cross-company: {cross}")