"""
src/ingestion/xbrl_facts.py   (UPDATED — adds gross_margin derived ratio)

Fetch EXACT financial numbers from the SEC's free XBRL companyfacts API.

WHAT'S NEW vs the first version
-------------------------------
- Adds a DERIVED concept "gross_margin" = GrossProfit / Revenue, computed
  per fiscal year, so questions like "Apple's gross margin in 2024" can be
  answered exactly instead of returning "insufficient evidence".
- format_value now handles percentages for *_margin concepts.

Endpoint: https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json
- Free, no key. 10 req/s. MANDATORY descriptive User-Agent w/ contact email.

Fetch ONCE at ingestion -> data/xbrl/<TICKER>.json + facts_lookup.json.
Query time NEVER calls the API.
"""

import os
import json
import time

import requests

SEC_USER_AGENT = os.getenv(
    "SEC_USER_AGENT",
    "Financial-Agentic-RAG harshitasaraogi01@gmail.com",
)

COMPANY_CIK = {
    "AAPL": "0000320193",
    "AMZN": "0001018724",
    "GOOGL": "0001652044",
    "MSFT": "0000789019",
    "TSLA": "0001318605",
}

# Direct GAAP concepts (first matching tag wins).
CONCEPTS = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ],
    "net_income": ["NetIncomeLoss"],
    "operating_income": ["OperatingIncomeLoss"],
    "gross_profit": ["GrossProfit"],
    "total_assets": ["Assets"],
    "total_liabilities": ["Liabilities"],
    "diluted_eps": ["EarningsPerShareDiluted"],
    "rnd_expense": ["ResearchAndDevelopmentExpense"],
    "operating_cash_flow": [
        "NetCashProvidedByUsedInOperatingActivities",
    ],
}

OUT_DIR = "data/xbrl"


def fetch_companyfacts(cik: str) -> dict:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    headers = {"User-Agent": SEC_USER_AGENT}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def extract_annual(facts: dict, gaap_tags: list) -> dict:
    """Return {fiscal_year: value} for the first GAAP tag that exists."""
    usgaap = facts.get("facts", {}).get("us-gaap", {})
    for tag in gaap_tags:
        if tag not in usgaap:
            continue
        units = usgaap[tag].get("units", {})
        for _unit, entries in units.items():
            out = {}
            for e in entries:
                if e.get("form") == "10-K" and e.get("fp") == "FY":
                    fy = e.get("fy")
                    val = e.get("val")
                    if fy is not None and val is not None:
                        out[int(fy)] = val
            if out:
                return out
    return {}


def build_lookup():
    os.makedirs(OUT_DIR, exist_ok=True)
    lookup = {}

    for ticker, cik in COMPANY_CIK.items():
        print(f"Fetching XBRL for {ticker} (CIK {cik}) ...")
        try:
            facts = fetch_companyfacts(cik)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        with open(os.path.join(OUT_DIR, f"{ticker}.json"), "w",
                  encoding="utf-8") as f:
            json.dump(facts, f)

        # Direct concepts
        per_concept_annual = {}
        for concept, tags in CONCEPTS.items():
            annual = extract_annual(facts, tags)
            per_concept_annual[concept] = annual
            for fy, val in annual.items():
                lookup[f"{ticker}|{concept}|{fy}"] = val

        # DERIVED: gross_margin = gross_profit / revenue (per year)
        gp = per_concept_annual.get("gross_profit", {})
        rev = per_concept_annual.get("revenue", {})
        for fy in set(gp) & set(rev):
            if rev[fy]:
                lookup[f"{ticker}|gross_margin|{fy}"] = gp[fy] / rev[fy]

        time.sleep(0.5)

    with open(os.path.join(OUT_DIR, "facts_lookup.json"), "w",
              encoding="utf-8") as f:
        json.dump(lookup, f, indent=2)

    print(f"\nWrote {len(lookup)} facts to {OUT_DIR}/facts_lookup.json")
    return lookup


# ============================================================
# QUERY-TIME HELPERS (no network)
# ============================================================

_LOOKUP_CACHE = None


def _load_lookup():
    global _LOOKUP_CACHE
    if _LOOKUP_CACHE is None:
        path = os.path.join(OUT_DIR, "facts_lookup.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                _LOOKUP_CACHE = json.load(f)
        else:
            _LOOKUP_CACHE = {}
    return _LOOKUP_CACHE


def get_fact(company: str, concept: str, year: int):
    lookup = _load_lookup()
    return lookup.get(f"{company}|{concept}|{year}")


def format_value(concept: str, value) -> str:
    if value is None:
        return "N/A"
    if concept.endswith("_margin"):
        return f"{value*100:.1f}%"
    if concept == "diluted_eps":
        return f"${value:,.2f}"
    if abs(value) >= 1e9:
        return f"${value/1e9:,.2f} billion"
    if abs(value) >= 1e6:
        return f"${value/1e6:,.2f} million"
    return f"${value:,.0f}"


if __name__ == "__main__":
    build_lookup()
    print("\nSmoke test:")
    print("AAPL revenue 2024:",
          format_value("revenue", get_fact("AAPL", "revenue", 2024)))
    print("AAPL gross_margin 2024:",
          format_value("gross_margin", get_fact("AAPL", "gross_margin", 2024)))