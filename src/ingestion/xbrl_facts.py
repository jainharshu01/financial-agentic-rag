"""
src/ingestion/xbrl_facts.py   (ROBUST extraction — fixes missing-year gaps)

Fetch EXACT financial numbers from the SEC's free XBRL companyfacts API.

WHY THIS VERSION
----------------
The evaluation showed Microsoft / Tesla / Alphabet FY2024 revenue were NOT
being injected as XBRL facts, so the weak generator fell back to tables and
picked the wrong row (e.g. Tesla "deferred revenue" instead of total
revenue). Two bugs in the old extractor caused the gaps:

1. FIRST-TAG-WINS. extract_annual returned as soon as ONE candidate GAAP
   tag had any data. If a company's primary tag (e.g.
   RevenueFromContractWithCustomerExcludingAssessedTax) only covered older
   years, the fuller `Revenues` tag was never consulted. -> Now MERGES all
   candidate tags into one {year: value} map.

2. YEAR FROM THE `fy` FIELD. The companyfacts `fy` field is the fiscal year
   of the REPORT an entry appeared in, not necessarily the period the value
   covers, and it collides across the 3 comparative years in a 10-K. -> Now
   derives the fiscal year from the period END DATE (int(end[:4])), which is
   correct for all five companies (Apple FY ends Sep, MSFT Jun, the rest
   Dec). Annual flow items also require a ~year-long period so quarterly
   facts can't leak in.

IMPORTANT — YOU MUST REBUILD THE LOOKUP
---------------------------------------
This only changes how facts are EXTRACTED. The query-time lookup file
(data/xbrl/facts_lookup.json) is unchanged until you regenerate it:

    python -m src.ingestion.xbrl_facts

That step needs internet (SEC API). After it runs, re-run the evaluation.

SEGMENT METRICS (AWS / automotive / cloud) are NOT in these top-level
concepts — they live behind XBRL dimensional members and are out of scope
here; those numbers now rely on the table-boosted retrieval path instead.

Endpoint: https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json
- Free, no key. 10 req/s. MANDATORY descriptive User-Agent w/ contact email.
"""

import os
import json
import time

import requests

SEC_USER_AGENT = os.getenv(
    "SEC_USER_AGENT",
    "Financial-Agentic-RAG contact@example.com",
)

COMPANY_CIK = {
    "AAPL": "0000320193",
    "AMZN": "0001018724",
    "GOOGL": "0001652044",
    "MSFT": "0000789019",
    "TSLA": "0001318605",
}

# Direct GAAP concepts. All listed tags are now MERGED (not first-wins).
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


def _year_from_end(end: str):
    """Fiscal year ~= calendar year of the period end date."""
    try:
        return int(str(end)[:4])
    except (ValueError, TypeError):
        return None


def _is_annual_period(start, end) -> bool:
    """
    For flow items (revenue, income, cash flow) the entry must span ~a year
    so quarterly facts don't leak in. Instant items (assets/liabilities,
    EPS) have no `start` and pass through.
    """
    if not start:
        return True
    try:
        from datetime import date
        s = date.fromisoformat(str(start))
        e = date.fromisoformat(str(end))
        return (e - s).days >= 300
    except Exception:
        return True


def extract_annual(facts: dict, gaap_tags: list) -> dict:
    """
    Return {fiscal_year: value} MERGED across every candidate GAAP tag,
    using 10-K annual entries keyed by the period end-date year. When the
    same year appears more than once, the most recently FILED value wins.
    """
    usgaap = facts.get("facts", {}).get("us-gaap", {})
    best = {}  # year -> (filed_date_str, value)

    for tag in gaap_tags:
        if tag not in usgaap:
            continue
        units = usgaap[tag].get("units", {})
        for _unit, entries in units.items():
            for e in entries:
                if e.get("form") != "10-K":
                    continue
                if not _is_annual_period(e.get("start"), e.get("end")):
                    continue
                fy = _year_from_end(e.get("end"))
                val = e.get("val")
                if fy is None or val is None:
                    continue
                filed = e.get("filed", "")
                if fy not in best or filed >= best[fy][0]:
                    best[fy] = (filed, val)

    return {yr: v for yr, (f, v) in best.items()}


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

        per_concept_annual = {}
        for concept, tags in CONCEPTS.items():
            annual = extract_annual(facts, tags)
            per_concept_annual[concept] = annual
            for fy, val in annual.items():
                lookup[f"{ticker}|{concept}|{fy}"] = val
            print(f"  {concept:18s}: years {sorted(annual)}")

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
    for t in ("AAPL", "MSFT", "TSLA", "GOOGL", "AMZN"):
        print(f"{t} revenue 2024:",
              format_value("revenue", get_fact(t, "revenue", 2024)))