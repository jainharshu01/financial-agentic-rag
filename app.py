import streamlit as st
import time
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
    ["Baseline RAG", "Agentic RAG", "Compare Both"]
)

company = st.sidebar.selectbox(
    "Select Company",
    ["AAPL", "AMZN", "GOOGL", "MSFT", "TSLA"]
)

year = st.sidebar.selectbox(
    "Select Year",
    ["Auto",2023, 2024, 2025]
)

# Convert Auto -> None

if year == "Auto":
    year = None

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
                # BASELINE PIPELINE
                # ====================================================

                if pipeline_type == "Baseline RAG":

                    start_time = time.time()

                    response = baseline_answer(
                        question=question,
                        company=company,
                        year=year
                    )

                    baseline_time = round(
                        time.time() - start_time,
                        2
                    )

                    answer = response["answer"]
                    results = response["results"]

                # ====================================================
                # AGENTIC PIPELINE
                # ====================================================

                elif pipeline_type == "Agentic RAG":

                    start_time = time.time()

                    response = agentic_answer(
                        question=question,
                        company=company,
                        year=year
                    )

                    agentic_time = round(
                        time.time() - start_time,
                        2
                    )

                    answer = response["answer"]
                    results = response["results"]
                    strategy = response["strategy"]

                # ====================================================
                # COMPARE BOTH
                # ====================================================

                else:

                    baseline_start = time.time()

                    baseline_response = baseline_answer(
                        question=question,
                        company=company,
                        year=year
                    )

                    baseline_compare_time = round(
                        time.time() - baseline_start,
                        2
                    )

                    agentic_start = time.time()

                    agentic_response = agentic_answer(
                        question=question,
                        company=company,
                        year=year
                    )

                    agentic_compare_time = round(
                        time.time() - agentic_start,
                        2
                    )

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
            # DISPLAY RESULTS
            # ====================================================

            if pipeline_type == "Compare Both":

                col1, col2 = st.columns(2)

                with col1:

                    st.subheader("📘 Baseline RAG")

                    st.write(baseline_response["answer"])

                    st.caption(
                        f"Retry Used: {baseline_response['retry_used']}"
                    )

                    st.info(
                        f"Response Time: {baseline_compare_time} sec"
                    )

                with col2:

                    st.subheader("🤖 Agentic RAG")

                    st.write(agentic_response["answer"])
                    
                    st.caption(
                        f"Retry Used: {agentic_response['retry_used']}"
                    )

                    st.info(
                        f"Response Time: {agentic_compare_time} sec"
                    )

                st.success("Comparison completed!")

            else:

                st.subheader("📄 Generated Answer")

                st.success("Analysis completed successfully!")

                st.write(answer)

                if pipeline_type == "Baseline RAG":

                    st.info(f"Response Time: {baseline_time} seconds")

                elif pipeline_type == "Agentic RAG":

                    st.info(f"Response Time: {agentic_time} seconds")

                if "retry_used" in response:

                    if response["retry_used"]:

                        st.warning("Retry retrieval was triggered.")

                    else:

                        st.success("No retrieval retry needed.")

                    # ====================================================
                    # RETRIEVED SOURCES
                    # ====================================================

                    st.subheader("📚 Retrieved Sources")

                    documents = results["documents"][0]
                    metadatas = results["metadatas"][0]
                    distances = results["distances"][0]

                    for i in range(len(documents)):

                        metadata = metadatas[i]
                        distance = distances[i]
                        similarity_percentage = round((1 - distance) * 100, 2)

                        with st.expander(
                            f"Source {i+1} | "
                            f"{metadata['section']} | "
                            f"Similarity: {similarity_percentage}%"
                        ):

                            st.markdown(f"""
                            **Company:** {metadata['company']}

                            **Year:** {metadata['year']}

                            **Section:** {metadata['section']}

                            **Similarity Score:** {similarity_percentage}%
                            """)

                            st.write(documents[i])
                        
        # ========================================================
        # ERROR HANDLING
        # ========================================================

        except Exception as e:

            st.error("An error occurred while processing the query.")

            st.exception(e)