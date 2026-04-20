from sec_edgar_downloader import Downloader

# SEC requires identification
dl = Downloader("Harshita Saraogi", "harshitasaraogi01@gmail.com", "data/raw_filings")

# 5 companies, 10-K filings, last 3 years
companies = ["AAPL", "MSFT", "TSLA", "AMZN", "GOOGL"]

for company in companies:
    print(f"Downloading 10-K filings for {company}...")
    try:
        dl.get("10-K", company, limit=3)
        print(f"  Done: {company}")
    except Exception as e:
        print(f"  Error for {company}: {e}")

print("\nAll downloads complete!")