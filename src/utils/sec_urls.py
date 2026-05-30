"""
src/utils/sec_urls.py

Builds direct SEC EDGAR URLs from filing metadata.

SEC URL pattern:
    https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no_dashes}/{accession_with_dashes}-index.htm

The accession number is your filing_id (e.g., '0000320193-24-000123').
"""

# ============================================================
# CIK LOOKUP — Central Index Key per company
# ============================================================

COMPANY_CIK = {
    "AAPL": "320193",       # Apple Inc.
    "AMZN": "1018724",      # Amazon.com Inc.
    "GOOGL": "1652044",     # Alphabet Inc.
    "MSFT": "789019",       # Microsoft Corporation
    "TSLA": "1318605",      # Tesla Inc.
}


def get_filing_url(company: str, filing_id: str) -> str:
    """
    Build the SEC EDGAR filing index URL.

    Args:
        company: Ticker (e.g., 'AAPL')
        filing_id: Accession number (e.g., '0000320193-24-000123')

    Returns:
        Full URL string, or empty string if inputs are invalid.

    Example:
        >>> get_filing_url('AAPL', '0000320193-24-000123')
        'https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/0000320193-24-000123-index.htm'
    """
    if not company or not filing_id:
        return ""

    cik = COMPANY_CIK.get(company.upper())
    if not cik:
        return ""

    # Strip dashes for path component
    accession_no_dashes = filing_id.replace("-", "")

    return (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{cik}/{accession_no_dashes}/{filing_id}-index.htm"
    )


def get_company_filings_url(company: str) -> str:
    """Build URL to browse all filings for a company."""
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
    test_cases = [
        ("AAPL", "0000320193-24-000123"),
        ("TSLA", "0001628280-25-000003"),
        ("MSFT", "0000950170-24-087843"),
    ]
    for company, filing_id in test_cases:
        url = get_filing_url(company, filing_id)
        print(f"{company} {filing_id}")
        print(f"  → {url}")
        print()