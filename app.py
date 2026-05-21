import streamlit as st

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Financial Agentic RAG",
    layout="wide"
)

# ============================================================
# TITLE
# ============================================================

st.title("📊 Financial Agentic RAG System")

st.markdown("""
Ask questions about SEC filings using:

- Baseline Static RAG
- Agentic RAG
""")

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Settings")

pipeline_type = st.sidebar.selectbox(
    "Choose Pipeline",
    ["Baseline RAG", "Agentic RAG"]
)

company = st.sidebar.selectbox(
    "Select Company",
    ["AAPL", "AMZN", "GOOGL", "MSFT", "TSLA"]
)

year = st.sidebar.selectbox(
    "Select Year",
    [2023, 2024, 2025]
)

# ============================================================
# QUERY INPUT
# ============================================================

question = st.text_area(
    "Enter your financial question:",
    height=120
)

# ============================================================
# BUTTON
# ============================================================

if st.button("Generate Answer"):

    st.info("Processing query...")

    st.write("### Selected Configuration")

    st.write(f"Pipeline: {pipeline_type}")
    st.write(f"Company: {company}")
    st.write(f"Year: {year}")

    st.write("### User Question")

    st.write(question)

    st.success("UI working successfully!")