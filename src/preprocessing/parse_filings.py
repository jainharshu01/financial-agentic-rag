import os
from bs4 import BeautifulSoup
import re

def extract_text_from_filing(file_path):
    """Read raw SEC filing and extract clean text."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    
    # Extract only the HTML document part (between <DOCUMENT> tags)
    # SEC filings have multiple documents, we want the 10-K one
    documents = re.findall(r"<DOCUMENT>(.*?)</DOCUMENT>", content, re.DOTALL)
    
    # Find the 10-K document (usually the first and largest one)
    main_doc = ""
    for doc in documents:
        if "<TYPE>10-K" in doc[:200]:
            main_doc = doc
            break
    
    # If no 10-K type found, use the largest document
    if not main_doc and documents:
        main_doc = max(documents, key=len)
    
    # If still nothing, use full content
    if not main_doc:
        main_doc = content
    
    # Parse HTML and extract text
    soup = BeautifulSoup(main_doc, "lxml")
    text = soup.get_text(separator="\n")
    
    # Clean up the text
    text = re.sub(r"\n{3,}", "\n\n", text)  # Remove excessive newlines
    text = re.sub(r" {2,}", " ", text)       # Remove excessive spaces
    text = re.sub(r"\t+", " ", text)         # Remove tabs
    
    # Remove empty lines that only have spaces
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(line for line in lines if line)
    
    return text


def process_all_filings():
    """Process all downloaded filings."""
    raw_base = "data/raw_filings/sec-edgar-filings"
    output_base = "data/processed"
    
    os.makedirs(output_base, exist_ok=True)
    
    for company in os.listdir(raw_base):
        company_path = os.path.join(raw_base, company, "10-K")
        
        if not os.path.exists(company_path):
            print(f"No 10-K folder for {company}, skipping.")
            continue
        
        for filing_folder in os.listdir(company_path):
            filing_path = os.path.join(company_path, filing_folder, "full-submission.txt")
            
            if not os.path.exists(filing_path):
                print(f"  No file found: {filing_path}")
                continue
            
            print(f"Processing {company} - {filing_folder}...")
            
            # Extract clean text
            clean_text = extract_text_from_filing(filing_path)
            
            # Save clean text
            output_file = os.path.join(output_base, f"{company}_{filing_folder}.txt")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(clean_text)
            
            print(f"  Saved: {output_file} ({len(clean_text)} characters)")


if __name__ == "__main__":
    process_all_filings()
    print("\nAll filings processed!")