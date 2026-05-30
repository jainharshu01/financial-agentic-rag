"""
src/ingestion/parse_filings.py   (REPLACES the old version)

Table-aware SEC 10-K parser.

WHY THIS CHANGED
----------------
The previous parser ran soup.get_text() over the whole filing, which
flattened every financial table into an orphaned vertical column of
numbers. "Net sales 391,035" became two unrelated lines, destroying the
row/column relationships that numeric queries depend on. That was the
root cause of the "insufficient evidence" failures on numeric questions.

WHAT THIS DOES NOW
------------------
1. Extracts each <table> with pandas.read_html and serializes it to a
   Markdown pipe table (headers preserved, relationships intact).
2. Replaces each table in the DOM with a unique placeholder token.
3. Flattens the remaining prose as before.
4. Re-injects the serialized Markdown tables at their placeholders.
5. Writes a SECOND file (<name>.tables.json) holding the tables alone,
   so the chunker can emit them as dedicated "table" chunks.

The text output therefore contains readable Markdown tables inline, and
the chunker can also pick up the structured tables list separately.
"""

import os
import re
import json
import warnings

import pandas as pd
from bs4 import BeautifulSoup

warnings.simplefilter(action="ignore", category=FutureWarning)


# ============================================================
# TABLE SERIALIZATION
# ============================================================

def _df_to_markdown(df: pd.DataFrame) -> str:
    """Serialize a DataFrame to a compact Markdown pipe table."""
    df = df.fillna("")
    # Flatten multi-index headers if present.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            " ".join(str(x) for x in col if str(x) != "nan").strip()
            for col in df.columns
        ]
    cols = [str(c).strip() for c in df.columns]

    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        cells = [str(v).strip().replace("\n", " ") for v in row.tolist()]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _looks_financial(md: str) -> bool:
    """Heuristic: keep tables that actually contain numbers."""
    digits = sum(c.isdigit() for c in md)
    return digits >= 8  # at least a few numeric cells


def extract_tables(soup: BeautifulSoup):
    """
    Replace each <table> in the soup with a placeholder and return a list
    of serialized Markdown tables in document order.

    Returns:
        (placeholders_map, tables_list)
        placeholders_map: {token: markdown}
        tables_list:      [markdown, ...] in order
    """
    placeholders = {}
    tables_list = []

    for idx, table in enumerate(soup.find_all("table")):
        token = f"@@TABLE_{idx}@@"
        try:
            # read_html needs a string; wrap the single table.
            dfs = pd.read_html(str(table))
            if not dfs:
                table.replace_with("")
                continue
            md = _df_to_markdown(dfs[0])
        except Exception:
            # Fallback: cell-by-cell text join, still better than flatten.
            rows = []
            for tr in table.find_all("tr"):
                cells = [
                    td.get_text(" ", strip=True)
                    for td in tr.find_all(["td", "th"])
                ]
                if cells:
                    rows.append(" | ".join(cells))
            md = "\n".join(rows)

        if md and _looks_financial(md):
            placeholders[token] = md
            tables_list.append(md)
            table.replace_with(token)
        else:
            table.replace_with("")  # drop noise tables (layout spacers)

    return placeholders, tables_list


# ============================================================
# MAIN EXTRACTION
# ============================================================

def extract_text_from_filing(file_path):
    """
    Read a raw SEC filing and return (clean_text, tables_list).

    clean_text contains Markdown tables inline at their original spots.
    tables_list is the same tables as standalone Markdown strings.
    """
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    documents = re.findall(r"<DOCUMENT>(.*?)</DOCUMENT>", content, re.DOTALL)

    main_doc = ""
    for doc in documents:
        if "<TYPE>10-K" in doc[:200]:
            main_doc = doc
            break
    if not main_doc and documents:
        main_doc = max(documents, key=len)
    if not main_doc:
        main_doc = content

    soup = BeautifulSoup(main_doc, "lxml")

    # 1) Pull tables out FIRST, leaving placeholders behind.
    placeholders, tables_list = extract_tables(soup)

    # 2) Flatten the remaining prose.
    text = soup.get_text(separator="\n")

    # 3) Clean whitespace.
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\t+", " ", text)
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(line for line in lines if line)

    # 4) Re-inject Markdown tables at their placeholder tokens.
    for token, md in placeholders.items():
        text = text.replace(token, f"\n\n[TABLE]\n{md}\n[/TABLE]\n\n")

    return text, tables_list


# ============================================================
# PROCESS ALL FILES
# ============================================================

def process_all_filings():
    raw_base = "data/raw_filings/sec-edgar-filings"
    output_base = "data/processed"
    os.makedirs(output_base, exist_ok=True)

    for company in os.listdir(raw_base):
        company_path = os.path.join(raw_base, company, "10-K")
        if not os.path.exists(company_path):
            print(f"No 10-K folder for {company}, skipping.")
            continue

        for filing_folder in os.listdir(company_path):
            filing_path = os.path.join(
                company_path, filing_folder, "full-submission.txt"
            )
            if not os.path.exists(filing_path):
                print(f"  No file found: {filing_path}")
                continue

            print(f"Processing {company} - {filing_folder}...")
            clean_text, tables_list = extract_text_from_filing(filing_path)

            out_txt = os.path.join(
                output_base, f"{company}_{filing_folder}.txt"
            )
            with open(out_txt, "w", encoding="utf-8") as f:
                f.write(clean_text)

            out_tables = os.path.join(
                output_base, f"{company}_{filing_folder}.tables.json"
            )
            with open(out_tables, "w", encoding="utf-8") as f:
                json.dump(tables_list, f, indent=2)

            print(
                f"  Saved: {out_txt} ({len(clean_text)} chars, "
                f"{len(tables_list)} tables)"
            )


if __name__ == "__main__":
    process_all_filings()
    print("\nAll filings processed (table-aware)!")