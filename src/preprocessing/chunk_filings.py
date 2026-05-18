import os
import re
import json

def extract_year_from_filing_id(filing_id):
    """
    Extract filing year from SEC filing ID.
    
    Example:
    0000320193-24-000123 -> 2024
    """
    
    try:
        parts = filing_id.split("-")
        short_year = parts[1]

        # Convert 24 -> 2024
        year = int(f"20{short_year}")

        return year

    except Exception:
        return None
# ============================================================
# METHOD 1: Fixed-size chunking
# ============================================================
def chunk_fixed(text, company, filing_id, chunk_size=500, overlap=100):
    """Split text into fixed-size word chunks with overlap."""
    words = text.split()
    chunks = []
    chunk_num = 0

    year = extract_year_from_filing_id(filing_id)

    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        if len(chunk_words) < 50:  # Skip very small leftover chunks
            continue

        chunk_text = " ".join(chunk_words)
        chunks.append({
            "chunk_id": f"{company}_{filing_id}_fixed_{chunk_num}",
            "company": company,
            "year": year,
            "filing_id": filing_id,
            "chunk_method": "fixed",
            "section": "unknown",
            "text": chunk_text
        })
        chunk_num += 1

    return chunks


# ============================================================
# METHOD 2: Section-based chunking
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
    """Extract major sections from filing text."""
    section_positions = []

    for section_name, pattern in SECTION_PATTERNS:
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if matches:
            # Use the LAST match (skip table of contents matches)
            match = matches[-1]
            section_positions.append((section_name, match.start()))

    # Sort by position
    section_positions.sort(key=lambda x: x[1])

    # Extract text for each section
    sections = {}
    for i, (name, start) in enumerate(section_positions):
        if i + 1 < len(section_positions):
            end = section_positions[i + 1][1]
        else:
            end = len(text)
        sections[name] = text[start:end]

    return sections


def chunk_by_section(text, company, filing_id, max_chunk_size=800):
    """Split text by section, then sub-chunk if section is too large."""
    sections = extract_sections(text)
    chunks = []
    chunk_num = 0

    year = extract_year_from_filing_id(filing_id)

    for section_name, section_text in sections.items():
        words = section_text.split()

        # If section is small enough, keep as one chunk
        if len(words) <= max_chunk_size:
            chunks.append({
                "chunk_id": f"{company}_{filing_id}_section_{chunk_num}",
                "company": company,
                "year": year,
                "filing_id": filing_id,
                "chunk_method": "section",
                "section": section_name,
                "text": section_text.strip()
            })
            chunk_num += 1
        else:
            # Sub-chunk large sections
            for i in range(0, len(words), max_chunk_size - 100):
                chunk_words = words[i:i + max_chunk_size]
                if len(chunk_words) < 50:
                    continue
                chunks.append({
                    "chunk_id": f"{company}_{filing_id}_section_{chunk_num}",
                    "company": company,
                    "year": year,
                    "filing_id": filing_id,
                    "chunk_method": "section",
                    "section": section_name,
                    "text": " ".join(chunk_words)
                })
                chunk_num += 1

    return chunks


# ============================================================
# PROCESS ALL FILES
# ============================================================
def process_all():
    processed_dir = "data/processed"
    chunks_dir = "data/chunks"
    os.makedirs(chunks_dir, exist_ok=True)

    all_fixed_chunks = []
    all_section_chunks = []

    for filename in sorted(os.listdir(processed_dir)):
        if not filename.endswith(".txt"):
            continue

        filepath = os.path.join(processed_dir, filename)
        company = filename.split("_")[0]
        filing_id = filename.replace(".txt", "").replace(f"{company}_", "")

        print(f"Chunking {filename}...")

        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        # Method 1: Fixed chunks
        fixed = chunk_fixed(text, company, filing_id)
        all_fixed_chunks.extend(fixed)
        print(f"  Fixed chunks: {len(fixed)}")

        # Method 2: Section-based chunks
        section = chunk_by_section(text, company, filing_id)
        all_section_chunks.extend(section)
        print(f"  Section chunks: {len(section)}")

    # Save all chunks
    with open(os.path.join(chunks_dir, "fixed_chunks.json"), "w", encoding="utf-8") as f:
        json.dump(all_fixed_chunks, f, indent=2)

    with open(os.path.join(chunks_dir, "section_chunks.json"), "w", encoding="utf-8") as f:
        json.dump(all_section_chunks, f, indent=2)

    print(f"\nTotal fixed chunks: {len(all_fixed_chunks)}")
    print(f"Total section chunks: {len(all_section_chunks)}")
    print("Saved to data/chunks/")


if __name__ == "__main__":
    process_all()