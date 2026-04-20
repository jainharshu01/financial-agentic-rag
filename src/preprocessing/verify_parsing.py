import os

file_path = "data/processed/AAPL_0000320193-24-000123.txt"

with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

sections = ["Item 1", "Risk Factors", "Item 7", "Financial Statements"]

print(f"File length: {len(text)} characters")
print()

for s in sections:
    pos = text.lower().find(s.lower())
    if pos != -1:
        print(f"FOUND: {s} at position {pos}")
        print(f"  Context: ...{text[pos:pos+100]}...")
        print()
    else:
        print(f"NOT FOUND: {s}")
        print()