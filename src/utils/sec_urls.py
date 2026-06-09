"""
src/utils/sec_urls.py  (UPDATED — direct document URLs)

Builds SEC EDGAR URLs from filing metadata.

WHAT CHANGED
------------
get_filing_url() previously returned the EDGAR INDEX page URL:
    .../0000320193-24-000123-index.htm

That page lists all exhibits and requires an extra click to reach the
actual 10-K document. This version returns the DIRECT 10-K document URL
for the 10 verified target filings, so the source link shown in the app
opens the 10-K immediately.

For any accession not in the table (future filings, amended filings, etc.)
the function gracefully falls back to the index URL — unchanged behaviour.

URL source: the verified SEC index links supplied in the evaluation brief.
"""

# ============================================================
# CIK LOOKUP — Central Index Key per company
# ============================================================

COMPANY_CIK = {
    "AAPL":  "320193",    # Apple Inc.
    "AMZN":  "1018724",   # Amazon.com Inc.
    "GOOGL": "1652044",   # Alphabet Inc.
    "MSFT":  "789019",    # Microsoft Corporation
    "TSLA":  "1318605",   # Tesla Inc.
}

# ============================================================
# DIRECT 10-K DOCUMENT URLs
# Keyed by accession number (with dashes).
# Source: the verified SEC filing links from the evaluation brief.
# ============================================================

DIRECT_DOC_URLS = {
    # ── FY2024 ────────────────────────────────────────────────────────────
    "0000320193-24-000123": (              # AAPL  aapl-20240928
        "https://www.sec.gov/Archives/edgar/data/320193/"
        "000032019324000123/aapl-20240928.htm"
    ),
    "0000950170-24-087843": (              # MSFT  msft-20240630
        "https://www.sec.gov/Archives/edgar/data/789019/"
        "000095017024087843/msft-20240630.htm"
    ),
    "0001018724-25-000004": (              # AMZN  amzn-20241231
        "https://www.sec.gov/Archives/edgar/data/1018724/"
        "000101872425000004/amzn-20241231.htm"
    ),
    "0001628280-25-003063": (              # TSLA  tsla-20241231
        "https://www.sec.gov/Archives/edgar/data/1318605/"
        "000162828025003063/tsla-20241231.htm"
    ),
    "0001652044-25-000014": (              # GOOGL goog-20241231
        "https://www.sec.gov/Archives/edgar/data/1652044/"
        "000165204425000014/goog-20241231.htm"
    ),
    # ── FY2023 ────────────────────────────────────────────────────────────
    "0000320193-23-000106": (              # AAPL  aapl-20230930
        "https://www.sec.gov/Archives/edgar/data/320193/"
        "000032019323000106/aapl-20230930.htm"
    ),
    "0000950170-23-035122": (              # MSFT  msft-20230630
        "https://www.sec.gov/Archives/edgar/data/789019/"
        "000095017023035122/msft-20230630.htm"
    ),
    "0001018724-24-000008": (              # AMZN  amzn-20231231
        "https://www.sec.gov/Archives/edgar/data/1018724/"
        "000101872424000008/amzn-20231231.htm"
    ),
    "0001628280-24-002390": (              # TSLA  tsla-20231231
        "https://www.sec.gov/Archives/edgar/data/1318605/"
        "000162828024002390/tsla-20231231.htm"
    ),
    "0001652044-24-000022": (              # GOOGL goog-20231231
        "https://www.sec.gov/Archives/edgar/data/1652044/"
        "000165204424000022/goog-20231231.htm"
    ),
}


def get_filing_url(company: str, filing_id: str) -> str:
    """
    Return the best available URL for this filing.

    Priority:
      1. Direct 10-K document URL  (from DIRECT_DOC_URLS — opens the filing
         immediately, no extra click needed).
      2. EDGAR index page URL       (fallback for any filing not in the table).

    Args:
        company:   Ticker (e.g. 'AAPL')
        filing_id: Accession number with dashes (e.g. '0000320193-24-000123')

    Returns:
        URL string, or empty string if inputs are invalid.
    """
    if not company or not filing_id:
        return ""

    # 1) Direct document URL — best experience for the user.
    direct = DIRECT_DOC_URLS.get(filing_id)
    if direct:
        return direct

    # 2) Fallback: EDGAR index page URL.
    cik = COMPANY_CIK.get(company.upper())
    if not cik:
        return ""
    accession_no_dashes = filing_id.replace("-", "")
    return (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{cik}/{accession_no_dashes}/{filing_id}-index.htm"
    )


def get_company_filings_url(company: str) -> str:
    """Build URL to browse all filings for a company on EDGAR."""
    cik = COMPANY_CIK.get(company.upper())
    if not cik:
        return ""
    return (
        f"https://www.sec.gov/cgi-bin/browse-edgar"
        f"?action=getcompany&CIK={cik}&type=10-K&dateb=&owner=include&count=40"
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    tests = [
        ("AAPL", "0000320193-24-000123"),   # FY2024 — should give direct doc URL
        ("TSLA", "0001628280-25-003063"),   # FY2024 — should give direct doc URL
        ("AMZN", "0001018724-24-000008"),   # FY2023 — should give direct doc URL
        ("MSFT", "0000789019-99-000001"),   # unknown — should fallback to index URL
    ]
    for company, filing_id in tests:
        url = get_filing_url(company, filing_id)
        label = "DIRECT" if "index.htm" not in url else "INDEX"
        print(f"\n{company} {filing_id}")
        print(f"  [{label}] {url}")