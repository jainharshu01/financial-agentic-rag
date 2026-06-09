"""
src/ingestion/download_filings.py  (UPDATED — targeted download)

Downloads EXACTLY the 10 verified FY2023 and FY2024 10-K filings.

WHY THIS CHANGED
----------------
The old version used limit=3, which downloaded FY2024, FY2023 AND FY2022
per company — adding 5 unnecessary FY2022 filings (15 total instead of 10).
Those extra filings wasted space, slowed build_vectorstore, and added noise
to retrieval.

This version uses a date-bounded window (after/before) for each target
filing so that exactly ONE 10-K is downloaded per call — the specific
filing verified in the evaluation dataset.

FILING DATES (from verified SEC index links)
--------------------------------------------
FY2024 filings:
  AAPL  aapl-20240928  filed 2024-11-01
  MSFT  msft-20240630  filed 2024-07-30
  AMZN  amzn-20241231  filed 2025-02-06
  TSLA  tsla-20241231  filed 2025-01-29
  GOOGL goog-20241231  filed 2025-02-04

FY2023 filings:
  AAPL  aapl-20230930  filed 2023-11-03
  MSFT  msft-20230630  filed 2023-07-25
  AMZN  amzn-20231231  filed 2024-02-01
  TSLA  tsla-20231231  filed 2024-01-29
  GOOGL goog-20231231  filed 2024-01-24

Run:
    python src/ingestion/download_filings.py
"""

from sec_edgar_downloader import Downloader

dl = Downloader(
    "Harshita Saraogi",
    "harshitasaraogi01@gmail.com",
    "data/raw_filings",
)

# (ticker, fiscal_year_label, after_date, before_date)
# Each window is chosen so that exactly ONE 10-K falls inside it.
TARGET_FILINGS = [
    # ── FY2024 ──────────────────────────────────────────────────────────────
    ("AAPL",  "FY2024", "2024-10-01", "2024-12-31"),   # filed 2024-11-01
    ("MSFT",  "FY2024", "2024-07-01", "2024-09-30"),   # filed 2024-07-30
    ("AMZN",  "FY2024", "2025-01-01", "2025-03-31"),   # filed 2025-02-06
    ("TSLA",  "FY2024", "2025-01-01", "2025-03-31"),   # filed 2025-01-29
    ("GOOGL", "FY2024", "2025-01-01", "2025-03-31"),   # filed 2025-02-04
    # ── FY2023 ──────────────────────────────────────────────────────────────
    ("AAPL",  "FY2023", "2023-10-01", "2023-12-31"),   # filed 2023-11-03
    ("MSFT",  "FY2023", "2023-07-01", "2023-09-30"),   # filed 2023-07-25
    ("AMZN",  "FY2023", "2024-01-01", "2024-03-31"),   # filed 2024-02-01
    ("TSLA",  "FY2023", "2024-01-01", "2024-03-31"),   # filed 2024-01-29
    ("GOOGL", "FY2023", "2024-01-01", "2024-03-31"),   # filed 2024-01-24
]


def main():
    print(f"Downloading {len(TARGET_FILINGS)} targeted 10-K filings "
          f"(FY2023 + FY2024 only)...\n")
    ok = 0
    for ticker, fy_label, after, before in TARGET_FILINGS:
        print(f"  {ticker} {fy_label}  ({after} → {before}) ... ", end="", flush=True)
        try:
            dl.get("10-K", ticker, limit=1, after=after, before=before)
            print("OK")
            ok += 1
        except Exception as e:
            print(f"ERROR: {e}")

    print(f"\nDone: {ok}/{len(TARGET_FILINGS)} filings downloaded.")
    print("Saved to: data/raw_filings/sec-edgar-filings/")


if __name__ == "__main__":
    main()