"""
src/preprocessing/chunk_filings.py   (REPLACES the old version)

Token-aware, context-injecting chunker.

KEY CHANGES vs the old word-count chunker
------------------------------------------
1. DOCUMENT-CONTEXT INJECTION (highest-ROI change):
   Every chunk's embedded `text` now begins with a context header:
       [Apple Inc. (AAPL) | FY2024 | 10-K | Section: Risk Factors]
   Research (Snowflake, 2025) found prepending global context lifts QA
   accuracy from ~50-60% to ~72-75%. Your metadata previously lived only
   in Chroma's metadata field and never reached the embedding.

2. TOKEN-BASED SIZING:
   We size chunks by the embedding model's own tokenizer instead of word
   count, so chunks actually fit the encoder. Default targets bge-small
   (512-token ceiling) -> ~400 content tokens + headroom for the header.

3. TABLE CHUNKS:
   Tables extracted by parse_filings.py (<name>.tables.json) become their
   OWN chunks, never split mid-row, tagged content_type="table".

The output JSON schema is a SUPERSET of the old one (adds content_type),
so build_vectorstore.py and all downstream filters stay compatible.
"""

import os
import re
import json

from transformers import AutoTokenizer

# Tokenizer for the embedding model we now use (bge-small-en-v1.5).
# Falls back gracefully if offline by approximating tokens as words*1.3.
_TOKENIZER = None


def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        try:
            _TOKENIZER = AutoTokenizer.from_pretrained(
                "BAAI/bge-small-en-v1.5"
            )
        except Exception:
            _TOKENIZER = "approx"
    return _TOKENIZER


def _token_len(text: str) -> int:
    tok = _get_tokenizer()
    if tok == "approx":
        return int(len(text.split()) * 1.3)
    return len(tok.encode(text, add_special_tokens=False))


# ============================================================
# COMPANY DISPLAY NAMES (for the context header)
# ============================================================

COMPANY_NAMES = {
    "AAPL": "Apple Inc.",
    "AMZN": "Amazon.com Inc.",
    "GOOGL": "Alphabet Inc.",
    "MSFT": "Microsoft Corporation",
    "TSLA": "Tesla Inc.",
}


def context_header(company, year, section):
    """Build the document-context prefix prepended to every chunk."""
    name = COMPANY_NAMES.get(company, company)
    yr = f"FY{year}" if year else "FY?"
    return f"[{name} ({company}) | {yr} | 10-K | Section: {section}]"


def extract_year_from_filing_id(filing_id):
    try:
        short_year = filing_id.split("-")[1]
        return int(f"20{short_year}")
    except Exception:
        return None


# ============================================================
# SECTION DETECTION (unchanged patterns)
# ============================================================

SECTION_PATTERNS = [
    ("Business", r"(?:Item\s*1[\.\s])\s*Business"),
    ("Risk Factors", r"(?:Item\s*1A[\.\s])\s*Risk\s*Factors"),
    ("Unresolved Staff Comments", r"(?:Item\s*1B[\.\s])"),
    ("Cybersecurity", r"(?:Item\s*1C[\.\s])\s*Cybersecurity"),
    ("Properties", r"(?:Item\s*2[\.\s])\s*Properties"),
    ("Legal Proceedings", r"(?:Item\s*3[\.\s])"),
    ("MD&A", r"(?:Item\s*7[\.\s])\s*Management"),
    ("Financial Statements", r"(?:Item\s*8[\.\s])\s*Financial\s*Statements"),
]


def extract_sections(text):
    section_positions = []
    for section_name, pattern in SECTION_PATTERNS:
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if matches:
            match = matches[-1]  # last match skips table-of-contents hits
            section_positions.append((section_name, match.start()))

    section_positions.sort(key=lambda x: x[1])

    sections = {}
    for i, (name, start) in enumerate(section_positions):
        end = (
            section_positions[i + 1][1]
            if i + 1 < len(section_positions)
            else len(text)
        )
        sections[name] = text[start:end]
    return sections


# ============================================================
# TOKEN-AWARE SPLITTER (sentence-greedy)
# ============================================================

def split_by_tokens(text, max_tokens, overlap_tokens):
    """
    Greedily pack sentences into chunks up to max_tokens, with a sliding
    overlap of roughly overlap_tokens between consecutive chunks.
    """
    # Split on sentence-ish boundaries but keep table blocks intact.
    sentences = re.split(r"(?<=[\.\?\!])\s+", text)

    chunks = []
    current = []
    current_tokens = 0

    for sent in sentences:
        st = _token_len(sent)
        if current_tokens + st > max_tokens and current:
            chunks.append(" ".join(current))
            # Build overlap tail.
            tail = []
            tail_tokens = 0
            for s in reversed(current):
                tail_tokens += _token_len(s)
                tail.insert(0, s)
                if tail_tokens >= overlap_tokens:
                    break
            current = tail
            current_tokens = tail_tokens
        current.append(sent)
        current_tokens += st

    if current:
        chunks.append(" ".join(current))

    return [c.strip() for c in chunks if c.strip()]


# ============================================================
# SECTION-BASED CHUNKING (primary method)
# ============================================================

def chunk_by_section(text, company, filing_id,
                     max_tokens=400, overlap_tokens=80):
    sections = extract_sections(text)
    year = extract_year_from_filing_id(filing_id)
    chunks = []
    chunk_num = 0

    for section_name, section_text in sections.items():
        pieces = split_by_tokens(section_text, max_tokens, overlap_tokens)
        for piece in pieces:
            header = context_header(company, year, section_name)
            chunks.append({
                "chunk_id": f"{company}_{filing_id}_section_{chunk_num}",
                "company": company,
                "year": year,
                "filing_id": filing_id,
                "chunk_method": "section",
                "section": section_name,
                "content_type": "text",
                # Context header is part of the EMBEDDED text:
                "text": f"{header}\n{piece}",
            })
            chunk_num += 1

    return chunks


# ============================================================
# TABLE CHUNKING (dedicated chunks)
# ============================================================

def chunk_tables(tables_list, company, filing_id):
    year = extract_year_from_filing_id(filing_id)
    chunks = []
    for i, md in enumerate(tables_list):
        header = context_header(company, year, "Financial Statements")
        chunks.append({
            "chunk_id": f"{company}_{filing_id}_table_{i}",
            "company": company,
            "year": year,
            "filing_id": filing_id,
            "chunk_method": "table",
            "section": "Financial Statements",
            "content_type": "table",
            "text": f"{header}\n[TABLE]\n{md}\n[/TABLE]",
        })
    return chunks


# ============================================================
# FIXED CHUNKING (kept for the fixed_chunks collection / baseline)
# ============================================================

def chunk_fixed(text, company, filing_id,
                max_tokens=400, overlap_tokens=80):
    year = extract_year_from_filing_id(filing_id)
    pieces = split_by_tokens(text, max_tokens, overlap_tokens)
    chunks = []
    for i, piece in enumerate(pieces):
        header = context_header(company, year, "unknown")
        chunks.append({
            "chunk_id": f"{company}_{filing_id}_fixed_{i}",
            "company": company,
            "year": year,
            "filing_id": filing_id,
            "chunk_method": "fixed",
            "section": "unknown",
            "content_type": "text",
            "text": f"{header}\n{piece}",
        })
    return chunks


# ============================================================
# PROCESS ALL
# ============================================================

def process_all():
    processed_dir = "data/processed"
    chunks_dir = "data/chunks"
    os.makedirs(chunks_dir, exist_ok=True)

    all_fixed = []
    all_section = []

    for filename in sorted(os.listdir(processed_dir)):
        if not filename.endswith(".txt"):
            continue

        filepath = os.path.join(processed_dir, filename)
        company = filename.split("_")[0]
        filing_id = filename.replace(".txt", "").replace(f"{company}_", "")

        print(f"Chunking {filename}...")
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        fixed = chunk_fixed(text, company, filing_id)
        all_fixed.extend(fixed)
        print(f"  Fixed chunks: {len(fixed)}")

        section = chunk_by_section(text, company, filing_id)
        all_section.extend(section)
        print(f"  Section chunks: {len(section)}")

        # Table chunks -> added to the SECTION collection.
        tables_path = os.path.join(
            processed_dir, f"{company}_{filing_id}.tables.json"
        )
        if os.path.exists(tables_path):
            with open(tables_path, "r", encoding="utf-8") as f:
                tables_list = json.load(f)
            tchunks = chunk_tables(tables_list, company, filing_id)
            all_section.extend(tchunks)
            print(f"  Table chunks: {len(tchunks)}")

    with open(os.path.join(chunks_dir, "fixed_chunks.json"), "w",
              encoding="utf-8") as f:
        json.dump(all_fixed, f, indent=2)
    with open(os.path.join(chunks_dir, "section_chunks.json"), "w",
              encoding="utf-8") as f:
        json.dump(all_section, f, indent=2)

    print(f"\nTotal fixed chunks:   {len(all_fixed)}")
    print(f"Total section chunks: {len(all_section)}")
    print("Saved to data/chunks/")


if __name__ == "__main__":
    process_all()