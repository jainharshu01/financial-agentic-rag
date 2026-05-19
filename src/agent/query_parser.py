import re


def extract_years(question):
    """
    Extract years mentioned in query.
    
    Example:
    "Compare Apple risks between 2023 and 2024"
    
    Returns:
    [2023, 2024]
    """

    years = re.findall(r"\b20\d{2}\b", question)

    years = [int(y) for y in years]

    return years


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    questions = [

        "Compare Apple risks between 2023 and 2024",

        "What was Tesla revenue in 2024?",

        "Compare Microsoft's revenue from 2022 and 2023",

        "Summarize Amazon outlook"
    ]

    for q in questions:

        years = extract_years(q)

        print("\nQuestion:", q)
        print("Extracted Years:", years)