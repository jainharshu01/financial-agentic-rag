"""
src/preprocessing/chunk_filings.py   (SECTION-BOUNDARY-CORRECTED version)

Token-aware, context-injecting chunker.

WHAT CHANGED IN THIS VERSION
----------------------------
The previous section detector picked, for each Item pattern, the LAST regex
match (`max(matches, key=start)`). That single heuristic broke in two ways:

  1. MSFT FY2024 "Risk Factors" collapsed to ONE chunk because the only
     `Item 1A ... Risk Factors` match in the document was the Table-of-Contents
     entry ("Item 1A. Risk Factors .... 20"). The real heading is formatted as a
     bare "RISK FACTORS" line that the Item-prefixed pattern never matched, so
     the TOC entry won by default.

  2. AMZN FY2024 "MD&A" collapsed to ONE chunk because a LATE in-prose
     cross-reference ("Item 7 of Part II ... Management's Discussion ...") was
     the last match, so `max(start)` selected the cross-reference instead of the
     real heading.

Both are symptoms of the same disease: matches are not all real headings. A
10-K is full of (a) a Table of Contents that lists every Item with a page
number, and (b) in-prose cross-references ("see Item 1A", "Item 7 of Part II").

ROOT-CAUSE FIX
--------------
Instead of "last match wins", every candidate match is now CLASSIFIED and only
genuine headings survive. We then take the FIRST surviving candidate (the real
heading always precedes any cross-reference to it in the body). Rejection rules:

  * In a [TABLE]...[/TABLE] block  -> TOC-as-table / financial-table noise.
    (The parser serialises the TOC table and re-injects it; this single rule
     removes almost all of the "77 chunks beginning with TOC references".)
  * Immediately followed by a page number  -> inline TOC entry.
  * Followed within a few lines by a bare page-number line AND another Item
    heading  -> inline TOC list block.
  * Preceded by cross-reference words ("see", "refer", "pursuant to", ...).
  * The matched text itself contains "of Part" / "in Part" / "under Part"
    -> cross-reference (this is exactly the AMZN "Item 7 of Part II" case).

If the Item-prefixed pattern yields NO surviving candidate (the MSFT case), we
fall back to BARE heading patterns ("RISK FACTORS" on its own line, etc.) run
under the same rejection rules. This recovers headings that don't sit next to
their "Item 1A." label.

The year / table / fixed-chunk logic is UNCHANGED from the fiscal-year fix.
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

# Calendar MONTH each company's fiscal year ends in.
COMPANY_FYE_MONTH = {
    "AAPL": 9,    # last Saturday of September
    "MSFT": 6,    # June 30
    "AMZN": 12,   # December 31
    "GOOGL": 12,  # December 31
    "TSLA": 12,   # December 31
}

# Verified accession -> FISCAL YEAR COVERED, taken from the official SEC
# index links. This is the source of truth for the target filings and
# overrides the heuristic below.
ACCESSION_FISCAL_YEAR = {
    # ---- FY2024 ----
    "0000320193-24-000123": 2024,  # AAPL  aapl-20240928
    "0000950170-24-087843": 2024,  # MSFT  msft-20240630
    "0001018724-25-000004": 2024,  # AMZN  amzn-20241231
    "0001628280-25-003063": 2024,  # TSLA  tsla-20241231
    "0001652044-25-000014": 2024,  # GOOGL goog-20241231
    # ---- FY2023 ----
    "0000320193-23-000106": 2023,  # AAPL  aapl-20230930
    "0000950170-23-035122": 2023,  # MSFT  msft-20230630
    "0001018724-24-000008": 2023,  # AMZN  amzn-20231231
    "0001628280-24-002390": 2023,  # TSLA  tsla-20231231
    "0001652044-24-000022": 2023,  # GOOGL goog-20231231
}


def context_header(company, year, section):
    """Build the document-context prefix prepended to every chunk."""
    name = COMPANY_NAMES.get(company, company)
    yr = f"FY{year}" if year else "FY?"
    return f"[{name} ({company}) | {yr} | 10-K | Section: {section}]"


def extract_year_from_filing_id(filing_id, company=None):
    """
    Return the FISCAL YEAR COVERED by the filing, NOT the submission year.

    Resolution order:
      1. Exact accession -> fiscal-year override (preferred, verified).
      2. Fiscal-calendar heuristic: December-year-end companies file the
         NEXT calendar year, so fiscal_year = submission_year - 1; companies
         whose fiscal year ends before December file in-year, so
         fiscal_year = submission_year.
    """
    if not filing_id:
        return None

    # 1) Exact override for known target filings.
    if filing_id in ACCESSION_FISCAL_YEAR:
        return ACCESSION_FISCAL_YEAR[filing_id]

    # 2) Heuristic from submission year + company fiscal calendar.
    try:
        submission_year = int(f"20{filing_id.split('-')[1]}")
    except Exception:
        return None

    fye_month = COMPANY_FYE_MONTH.get((company or "").upper())
    if fye_month is None:
        # Unknown company: fall back to old behaviour (best effort).
        return submission_year
    if fye_month == 12:
        return submission_year - 1
    return submission_year


# ============================================================
# SECTION DETECTION  (order-constrained spine reconstruction)
# ============================================================
# The previous detector matched each section independently and trusted the
# first surviving regex hit, then sliced [start_i, start_{i+1}) blindly. On the
# real filings that produced catastrophic mislabelling: cross-reference lines
# ("see Item 7, ...") and forward-looking enumerations were picked as headings
# (AAPL/GOOGL), Amazon's table-banner headings were thrown away, Microsoft's
# vertically letter-split headings ("ITEM 1A. RIS\nK FACTORS") never matched,
# and the last section ran to EOF swallowing all back-matter.
#
# This version reconstructs the canonical 10-K Item spine:
#   * Candidates come from BOTH prose ("Item 1A. Risk Factors ...") and from
#     table-banner rows (Amazon renders headings as "| Item 1A. | Risk Factors |").
#   * Titles are matched by WHITESPACE-STRIPPED prefix, so any vertical split
#     ("RIS\nK", "FINANCI AL") or nbsp run is absorbed automatically.
#   * Non-headings are rejected: TOC rows (trailing/embedded page numbers),
#     in-prose candidates inside [TABLE] blocks, multi-Item enumeration lines,
#     "Item N of Part ..." references, and same-line cross-references.
#   * Sections are then selected in canonical order (1, 1A, 1B, 1C, 2, 3, 7, 8),
#     each taking the EARLIEST survivor whose position is after the previously
#     accepted section -- a detected Item 7 before Item 3 is structurally
#     impossible and is discarded.
#   * An Item 9 / 9A sentinel caps "Financial Statements" so governance/exhibit
#     back-matter is not swallowed.

# (rank, item-key, canonical section name, cleaned title prefix)
SECTIONS = [
    (1.0, "1",  "Business",                  "business"),
    (1.1, "1A", "Risk Factors",              "riskfactors"),
    (1.2, "1B", "Unresolved Staff Comments", "unresolvedstaffcomments"),
    (1.3, "1C", "Cybersecurity",             "cybersecurity"),
    (2.0, "2",  "Properties",                "properties"),
    (3.0, "3",  "Legal Proceedings",         "legalproceedings"),
    (7.0, "7",  "MD&A",                      "managementsdiscussion"),
    (8.0, "8",  "Financial Statements",      "financialstatements"),
    # Boundary-only sentinel: the first Item 9 / 9A heading after Item 8 ends
    # the financial statements. Everything after it (Items 9-16: controls,
    # governance, executive comp, exhibits, signatures) is back-matter that must
    # NOT be absorbed into "Financial Statements". Dropped before chunking.
    (9.0, "9",  "_BACKMATTER",               "changesinand"),
    (9.0, "9A", "_BACKMATTER",               "controlsandprocedures"),
]
SEC_BY_KEY = {key: (rank, name, title) for rank, key, name, title in SECTIONS}

# Major sections we expect to be substantial; used by the audit warning.
MAJOR_SECTIONS = ("Business", "Risk Factors", "MD&A", "Financial Statements")

ITEM_TOKEN = re.compile(r"\bItem\s*(\d{1,2})([A-Ca-c])?\b", re.I)
TABLE_BLOCK = re.compile(r"\[TABLE\](.*?)\[/TABLE\]", re.DOTALL)

# Cross-reference words that, on the SAME line before an Item token, mark it as
# a reference rather than a heading.
_XREF_WORDS = (
    "see ", "see,", "refer", "pursuant", "described in", "set forth",
    "discussed in", "contained in", "incorporated", "included in",
    "presented in", "described under", "described above", "described below",
)


def _clean(s):
    """Lower-case and drop every non-alphanumeric char.

    This is the key normaliser: it makes 'RIS\\nK FACTORS', 'FINANCI AL
    STATEMENTS', 'Item\\xa01A.' and 'Risk Factors' all comparable, so vertical
    letter-splits and nbsp runs no longer defeat title matching.
    """
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _line_of(text, pos):
    a = text.rfind("\n", 0, pos)
    b = text.find("\n", pos)
    if b == -1:
        b = len(text)
    return text[a + 1:b]


def _trailing_pagenum(line):
    """TOC entry: a line ending in (leader/space) + a small page number."""
    return re.search(r"[\s.\u2026\u00b7\-\u2013\u2014]\s*\d{1,4}\s*$", line) is not None


def _xref_before(text, start):
    """Cross-reference word on the SAME line before the Item token.

    Bounded to the current line so a previous sentence ending in a word like
    'described in' cannot reject a real heading that begins the next line.
    """
    ls = text.rfind("\n", 0, start)
    window = (text[ls + 1:start] if ls != -1 else text[:start]).lower()
    return any(w in window for w in _XREF_WORDS)


def _of_part_ref(text, end):
    """Reference forms ('Item 7 of Part II', 'Item 1 of this Form 10-K').

    The 'of Part'/'of this Form' connector always sits IMMEDIATELY after the
    Item number in a cross-reference, before any title -- so we inspect only a
    tight window. A real heading ('ITEM 3. LEGAL PROCEEDINGS For a description
    of our ...') has its title there instead, and must not be rejected. We do
    NOT test ' of our'/' in part': those occur in legitimate heading prose, and
    genuine multi-item reference lists are removed by the enumeration rule.
    """
    low = text[end:end + 16].lower()
    return " of part" in low or " of this form" in low


def _row_has_pagenums(row):
    """A TOC banner row carries a page number in a numeric-only cell anywhere,
    e.g. '| Item 1. |  | Business |  | 4 |  | ...'. A real body banner
    ('| Item 1A. | ... | Risk Factors |') has no numeric cell."""
    return re.search(r"\|\s*\d{1,4}\s*\|", row) is not None


def _table_spans(text):
    return [(m.start(), m.end()) for m in TABLE_BLOCK.finditer(text)]


def _in_spans(pos, spans):
    return any(a <= pos < b for a, b in spans)


def _prose_candidates(text):
    """In-prose 'Item N <Title>' candidates (title matched stripped-prefix)."""
    out = []
    for m in ITEM_TOKEN.finditer(text):
        key = m.group(1) + (m.group(2) or "").upper()
        sec = SEC_BY_KEY.get(key)
        if not sec:
            continue
        rank, name, title = sec
        after = _clean(text[m.end():m.end() + 160])
        if after.startswith(title):
            out.append([m.start(), m.end(), rank, name, key, "prose"])
    return out


def _banner_candidates(text):
    """Table-banner headings (Amazon renders each Item heading as a table row).

    A banner row contains an Item token followed by its title and NO page-number
    cell. The section start is emitted at the END of the table block, where the
    section's prose begins. TOC rows (which carry page numbers) are rejected.
    """
    out = []
    for tb in TABLE_BLOCK.finditer(text):
        block = tb.group(1)
        for row in block.split("\n"):
            if "|" not in row:
                continue
            mm = ITEM_TOKEN.search(row)
            if not mm:
                continue
            key = mm.group(1) + (mm.group(2) or "").upper()
            sec = SEC_BY_KEY.get(key)
            if not sec:
                continue
            rank, name, title = sec
            if title not in _clean(row[mm.end():]):
                continue
            if _row_has_pagenums(row):
                continue
            out.append([tb.end(), tb.end(), rank, name, key, "banner"])
            break  # at most one heading banner per table block
    return out


def find_section_starts(text):
    """Return [(section_name, start), ...] for the canonical Item spine.

    Includes a trailing ('_BACKMATTER', pos) sentinel when an Item 9/9A heading
    is found; extract_sections uses it to cap Financial Statements and then
    drops it.
    """
    cands = _prose_candidates(text) + _banner_candidates(text)
    table_spans = _table_spans(text)

    survivors = []
    for s, e, rank, name, key, kind in cands:
        if kind == "prose":
            if _in_spans(s, table_spans):          # TOC rows / data-table noise
                continue
            line = _line_of(text, s)
            if len(ITEM_TOKEN.findall(line)) >= 2:  # multi-Item enumeration line
                continue
            if _trailing_pagenum(line):             # inline TOC entry
                continue
            if _of_part_ref(text, e):               # "Item 7 of Part II ..."
                continue
            if _xref_before(text, s):               # "see Item 1A ..."
                continue
        survivors.append((rank, s, name))

    by_sec = {}
    for rank, s, name in survivors:
        by_sec.setdefault((rank, name), []).append(s)

    # Canonical-order selection: each section takes the earliest survivor that
    # comes AFTER the previously accepted section (enforces 10-K Item order and
    # discards out-of-order cross-references).
    chosen = {}
    last = -1
    for (rank, name) in sorted(by_sec):
        pos = sorted(p for p in by_sec[(rank, name)] if p > last)
        if pos:
            chosen[name] = pos[0]
            last = pos[0]
    return sorted(chosen.items(), key=lambda kv: kv[1])


def _normalize(text):
    """Repair SEC vertical-letter heading artifacts (kept for chunk_fixed).

    Section DETECTION no longer relies on this -- the stripped-prefix title
    match handles arbitrary splits -- but it remains a harmless cosmetic pass
    for the baseline fixed-chunk text.
    """
    text = re.sub(r"B\s*\n\s*USINESS", "BUSINESS", text, flags=re.I)
    text = re.sub(r"RISK\s*\n\s*FACTORS", "RISK FACTORS", text, flags=re.I)
    text = re.sub(r"CY\s*\n\s*BERSECURITY", "CYBERSECURITY", text, flags=re.I)
    text = re.sub(r"PR\s*\n\s*OPERTIES", "PROPERTIES", text, flags=re.I)
    text = re.sub(r"LEGAL\s*\n\s*PROCEEDINGS", "LEGAL PROCEEDINGS", text, flags=re.I)
    text = re.sub(r"MANAGEMENT\S*\s*\n\s*DISCUSSION", "MANAGEMENT DISCUSSION",
                  text, flags=re.I)
    return text


def extract_sections(text):
    """Split the filing into {section_name: section_text} using the Item spine.

    Detection runs on the raw parsed text (the stripped-prefix matcher needs no
    pre-normalisation). The '_BACKMATTER' sentinel, if present, caps the
    Financial Statements span and is not emitted as its own section.
    """
    ordered = find_section_starts(text)

    sections = {}
    for i, (name, start) in enumerate(ordered):
        end = ordered[i + 1][1] if i + 1 < len(ordered) else len(text)
        if name == "_BACKMATTER":
            continue  # boundary only -- caps the previous section
        sections[name] = text[start:end]
    return sections




# ============================================================
# TOKEN-AWARE SPLITTER (sentence-greedy)  -- unchanged
# ============================================================

def split_by_tokens(text, max_tokens, overlap_tokens):
    """
    Greedily pack sentences into chunks up to max_tokens, with a sliding
    overlap of roughly overlap_tokens between consecutive chunks.
    """
    sentences = re.split(r"(?<=[\.\?\!])\s+", text)

    chunks = []
    current = []
    current_tokens = 0

    for sent in sentences:
        st = _token_len(sent)
        if current_tokens + st > max_tokens and current:
            chunks.append(" ".join(current))
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
    year = extract_year_from_filing_id(filing_id, company)
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
                "text": f"{header}\n{piece}",
            })
            chunk_num += 1

    return chunks


# ============================================================
# TABLE CHUNKING (dedicated chunks)  -- unchanged
# ============================================================

def chunk_tables(tables_list, company, filing_id):
    year = extract_year_from_filing_id(filing_id, company)
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
                max_tokens=400, overlap_tokens=80, drop_preamble=True):
    """
    Sliding-window chunker for the baseline collection.

    drop_preamble: trim the cover page + Table of Contents by starting at the
    first real section heading -- ONLY when that heading is found inside the
    first 30% of the document (a safety guard against over-trimming). This is
    what removes TOC chunks from the baseline; set False to reproduce the old
    full-document behaviour.
    """
    text = _normalize(text)
    year = extract_year_from_filing_id(filing_id, company)

    body = text
    if drop_preamble:
        ordered = find_section_starts(text)
        if ordered:
            first = ordered[0][1]
            if 0 < first < int(len(text) * 0.30):
                body = text[first:]

    pieces = split_by_tokens(body, max_tokens, overlap_tokens)
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

        # ---- per-section audit so boundary regressions are obvious ----------
        counts = {}
        for c in section:
            counts[c["section"]] = counts.get(c["section"], 0) + 1
        print(f"  Section chunks: {len(section)}")
        for name in sorted(counts, key=lambda n: -counts[n]):
            flag = ""
            if name in MAJOR_SECTIONS and counts[name] < 2:
                flag = "  <-- WARNING: likely boundary bug (too small)"
            print(f"      {name:<28} {counts[name]:>4}{flag}")
        for name in MAJOR_SECTIONS:
            if name not in counts:
                print(f"      {name:<28}    0  <-- WARNING: section MISSING")

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