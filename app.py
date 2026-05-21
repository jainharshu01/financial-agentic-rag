import streamlit as st

from src.retrieval.baseline_rag import baseline_answer
from src.agent.agentic_rag import agentic_answer
from src.agent.query_classifier import classify_query
from src.agent.router import build_retrieval_strategy

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
Analyze SEC filings using:

- Baseline Static RAG
- Agentic Adaptive RAG
""")

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Settings")

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
    height=150,
    placeholder="Example: Compare Apple's risks between 2023 and 2024"
)

# ============================================================
# GENERATE BUTTON
# ============================================================

if st.button("Generate Answer"):

    if question.strip() == "":

        st.warning("Please enter a question.")

    else:

        try:

            # ====================================================
            # LOADING SPINNER
            # ====================================================

            with st.spinner("Analyzing SEC filings..."):

                # ====================================================
                # QUERY CLASSIFICATION
                # ====================================================

                query_type = classify_query(question)

                strategy = build_retrieval_strategy(question)

                # ====================================================
                # DISPLAY QUERY ANALYSIS
                # ====================================================

                st.subheader("🧠 Query Analysis")

                col1, col2 = st.columns(2)

                with col1:
                    st.info(f"Pipeline: {pipeline_type}")

                with col2:
                    st.info(f"Query Type: {query_type}")

                # ====================================================
                # DISPLAY RETRIEVAL STRATEGY
                # ====================================================

                with st.expander("📌 Retrieval Strategy"):

                    st.json(strategy)

                # ====================================================
                # BASELINE PIPELINE
                # ====================================================

                if pipeline_type == "Baseline RAG":

                    answer = baseline_answer(
                        question=question,
                        company=company,
                        year=year
                    )

                # ====================================================
                # AGENTIC PIPELINE
                # ====================================================

                else:

                    answer = agentic_answer(
                        question=question,
                        company=company,
                        year=year
                    )

                # ====================================================
                # DISPLAY ANSWER
                # ====================================================

                st.subheader("📄 Generated Answer")

                st.success("Analysis completed successfully!")

                st.write(answer)

        # ========================================================
        # ERROR HANDLING
        # ========================================================

        except Exception as e:

            st.error("An error occurred while processing the query.")

            st.exception(e)