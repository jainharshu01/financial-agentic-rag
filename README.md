# 📊 Financial Agentic RAG System

[![HuggingFace Space](https://img.shields.io/badge/🤗-HuggingFace%20Space-yellow)](https://huggingface.co/spaces/jainharshu/financial-agentic-rag)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)

An **intelligent document retrieval and question-answering system** for analyzing SEC 10-K filings using **Retrieval-Augmented Generation (RAG)** with agentic enhancements. Compare baseline static retrieval against dynamic agentic routing, reranking, retry mechanisms, and answer validation.

🔗 **[Live Demo on HuggingFace Spaces](https://huggingface.co/spaces/jainharshu/financial-agentic-rag)**

---

## 🎯 Project Overview

This system processes **SEC 10-K filings** from five major companies (Apple, Amazon, Google, Microsoft, Tesla) across multiple years (2022-2025) and enables natural language queries with **grounded, cited answers**.

### Key Features

- **Dual Pipeline Architecture**: Side-by-side comparison of Baseline RAG vs. Agentic RAG
- **Intelligent Query Classification**: Automatically detects numeric, risk-based, descriptive, and comparative queries
- **Dynamic Section Routing**: Routes queries to the most relevant document sections (Business, Risk Factors, MD&A, Financial Statements)
- **Semantic Reranking**: Post-retrieval reranking using sentence-transformers for improved relevance
- **Retry Mechanism**: Automatically broadens search when initial retrieval yields low-confidence results
- **Answer Validation**: Checks for citation presence, unsupported claims, year coverage, and numeric accuracy
- **Source Grounding**: All answers include citations to specific source chunks with similarity scores
- **Automated Evaluation**: 40-question benchmark with metrics for section accuracy, citation rate, latency, and retrieval quality

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Query                               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                ┌────────────┴────────────┐
                │  Query Classifier        │
                │  (numeric/risk/desc/comp)│
                └────────────┬────────────┘
                             │
                ┌────────────┴────────────┐
                │  Section Router          │
                │  (Business/Risk/MD&A/FS) │
                └────────────┬────────────┘
                             │
         ┌───────────────────┴───────────────────┐
         │                                       │
    ┌────▼─────┐                          ┌─────▼─────┐
    │ Baseline │                          │  Agentic  │
    │   RAG    │                          │    RAG    │
    └────┬─────┘                          └─────┬─────┘
         │                                      │
         │  ┌─────────────────────────────┐    │
         │  │ ChromaDB Vectorstore        │◄───┤
         │  │ (all-MiniLM-L6-v2 embeddings)│    │
         │  └─────────────────────────────┘    │
         │                                      │
         │                              ┌───────▼────────┐
         │                              │  Reranker      │
         │                              │  (similarity)  │
         │                              └───────┬────────┘
         │                                      │
         │                              ┌───────▼────────┐
         │                              │  LLM (Groq)    │
         │                              │  llama-3.3-70b │
         │                              └───────┬────────┘
         │                                      │
         │                              ┌───────▼────────┐
         │                              │  Validator     │
         │                              │  (citations,   │
         │                              │   grounding)   │
         │                              └───────┬────────┘
         │                                      │
         │                              ┌───────▼────────┐
         │                              │  Retry Logic   │
         │                              │  (if needed)   │
         │                              └───────┬────────┘
         │                                      │
         └──────────────┬───────────────────────┘
                        │
                ┌───────▼────────┐
                │  Final Answer  │
                │  with Citations│
                └────────────────┘
```

---

## 📁 Repository Structure

```
financial-agentic-rag/
├── src/
│   ├── ingestion/
│   │   ├── download_filings.py      # SEC EDGAR downloader
│   │   └── parse_filings.py         # HTML → structured sections
│   ├── preprocessing/
│   │   ├── chunk_filings.py         # Semantic chunking (500-token overlapping)
│   │   ├── parse_filings.py         # Section extraction logic
│   │   └── verify_parsing.py        # Validation of parsed data
│   ├── retrieval/
│   │   ├── build_vectorstore.py     # ChromaDB initialization
│   │   ├── baseline_rag.py          # Static retrieval pipeline
│   │   ├── rag_pipeline.py          # Core retrieval logic
│   │   ├── reranker.py              # Post-retrieval reranking
│   │   └── test_retrieval.py        # Unit tests for retrieval
│   └── agent/
│       ├── query_classifier.py      # Query type detection
│       ├── router.py                # Section routing strategy
│       ├── agentic_rag.py           # Dynamic agentic pipeline
│       ├── answer_validator.py      # Answer quality checks
│       └── self_check.py            # Internal validation logic
├── evaluation/
│   ├── evaluation_questions.csv     # 40-question benchmark
│   ├── run_evaluation.py            # Automated evaluation pipeline
│   ├── metrics.py                   # Metric computation (Hit@k, section accuracy, etc.)
│   ├── dashboard.py                 # Streamlit evaluation dashboard
│   └── results.csv                  # Evaluation results (auto-generated)
├── data/
│   ├── vectorstore/                 # ChromaDB persistent storage (221MB)
│   ├── chunks/                      # Chunked filings (JSON)
│   └── processed/                   # Parsed section-level filings
├── app.py                           # Main Streamlit UI
├── requirements.txt                 # Python dependencies
├── runtime.txt                      # Python version for deployment
├── .env.example                     # Environment variable template
├── .gitignore
└── README.md
```

---


### Prerequisites

- **Python 3.11** (required for dependency compatibility)
- **Groq API Key** (free tier: [console.groq.com](https://console.groq.com))


## 🧪 Evaluation & Benchmarking

### Running the Benchmark

The evaluation pipeline tests both RAG approaches on **40 curated questions** across 4 categories:

```bash
python -m evaluation.run_evaluation
```

This generates `evaluation/results.csv` with metrics for:
- **Section Accuracy**: Did the router pick the right section?
- **Citation Rate**: Percentage of answers with proper citations
- **Retrieval Quality**: Average and top similarity scores
- **Latency**: Response time per pipeline
- **Retry Frequency**: How often the agentic pipeline needed a second attempt

### Viewing Results

```bash
streamlit run evaluation/dashboard.py
```

The dashboard includes:
- **Response time comparison** (baseline vs. agentic)
- **Similarity distributions** (avg vs. top)
- **Retry frequency** pie chart
- **Section accuracy by query type**
- **Citation presence rates**
- **Individual answer review** (side-by-side with gold answers)

### Sample Benchmark Results

| Metric                     | Baseline RAG | Agentic RAG |
|----------------------------|--------------|-------------|
| **Avg Response Time**      | 4.53s        | 6.33s       |
| **Section Accuracy**       | 85%          | 85%         |
| **Citation Rate**          | 55%          | 57%         |
| **Avg Similarity**         | 53%          | 52%         |
| **Retry Frequency**        | N/A          | 10%         |
| **Comparison Completeness**| 100%         | 87.5%       |

---

## 🔧 Technical Details

### Tech Stack

| Component           | Technology                          |
|---------------------|-------------------------------------|
| **Embeddings**      | `sentence-transformers/all-MiniLM-L6-v2` |
| **Vector Store**    | ChromaDB 0.5.x (persistent local)   |
| **LLM**             | Groq API (llama-3.3-70b-versatile)  |
| **Reranker**        | Bi-encoder cosine similarity        |
| **UI Framework**    | Streamlit 1.37.0                    |
| **Visualization**   | Plotly, custom CSS (DM Sans font)   |

### Data Pipeline

1. **Download** — SEC EDGAR API pulls HTML filings for 5 companies × 4 years = 20 filings
2. **Parse** — BeautifulSoup extracts 4 sections per filing (Business, Risk Factors, MD&A, Financial Statements)
3. **Chunk** — Semantic chunking with 500-token windows, 50-token overlap → ~1,200 chunks
4. **Embed** — SentenceTransformers generates 384-dim vectors
5. **Store** — ChromaDB persists embeddings + metadata (company, year, section, chunk_id)

### Query Processing

**Baseline RAG:**
1. Classify query type
2. Route to section (static mapping)
3. Retrieve top-k chunks (k=5)
4. Generate answer with LLM
5. Return answer + sources

**Agentic RAG:**
1. Classify query type
2. Route to section
3. Retrieve top-k chunks (k=3 initially)
4. **Rerank** chunks by semantic similarity
5. Generate answer with LLM
6. **Validate** answer (citations, grounding, year coverage)
7. **Retry** with broader k=8 if validation fails
8. Return answer + validation metadata + sources

---

## 📊 Retrieval Strategy

### Section Routing Logic

| Query Type      | Target Section(s)              | Example                                    |
|-----------------|--------------------------------|--------------------------------------------|
| **Numeric**     | Financial Statements, MD&A     | "What was Apple's revenue in 2023?"        |
| **Risk**        | Risk Factors                   | "What are Tesla's cybersecurity risks?"    |
| **Descriptive** | Business, MD&A                 | "Describe Google's search business model"  |
| **Comparative** | Multi-year retrieval (OR filter)| "Compare Amazon's risks 2023 vs 2024"     |

### Metadata Filtering

ChromaDB filters applied:
```python
# Single company, single year
{"company": "AAPL", "year": 2023, "section": "Risk Factors"}

# Comparative (multi-year)
{"company": "AAPL", "$or": [{"year": 2023}, {"year": 2024}]}
```

### Retry Logic

If initial retrieval yields:
- Avg similarity < 40%, OR
- Top similarity < 50%, OR
- Validator detects missing citations/years

→ Retry with `top_k=8` while **preserving original section filter**

---

## 📈 Performance Considerations

### Latency Breakdown

**Baseline RAG** (~4.5s):
- Retrieval: ~0.2s
- LLM inference: ~4.0s
- Formatting: ~0.3s

**Agentic RAG** (~6.3s):
- Retrieval: ~0.2s
- Reranking: ~0.4s
- LLM inference: ~4.0s
- Validation: ~0.2s
- Retry (when triggered): +1.5s
- Formatting: ~0.3s

### Optimization Opportunities

1. **Batch reranking** — vectorize all candidates at once instead of per-chunk
2. **Cache embeddings** — avoid re-embedding frequent queries
3. **Streaming LLM responses** — show partial answers while generating
4. **Parallel retrieval** — for multi-year comparative queries
5. **Cross-encoder reranking** — better accuracy than bi-encoder (but slower)

---

## 🐛 Known Limitations

1. **Citation detection regex** — model sometimes outputs "Source 1" without parentheses; metric expects "(Source 1)"
2. **Chunking granularity** — some numeric facts fall across chunk boundaries, causing "insufficient evidence" responses
3. **Section router accuracy** — ~15% of queries route to wrong section (especially descriptive queries mapped to MD&A instead of Business)
4. **Comparison completeness** — agentic pipeline sometimes retrieves only one year in multi-year queries despite OR filter
5. **No conversational memory** — each query is stateless (could add with `st.session_state`)

---

## 🛠️ Development

### Adding New Companies

1. Update `COMPANIES` list in `src/ingestion/download_filings.py`
2. Run download and parsing:
   ```bash
   python -m src.ingestion.download_filings
   python -m src.preprocessing.chunk_filings
   ```
3. Rebuild vectorstore:
   ```bash
   python -m src.retrieval.build_vectorstore
   ```

### Modifying Chunking Strategy

Edit `src/preprocessing/chunk_filings.py`:
- `chunk_size`: Default 500 tokens
- `chunk_overlap`: Default 50 tokens
- Adjust based on your use case (larger chunks = more context, fewer total chunks)

### Extending Evaluation

Add questions to `evaluation/evaluation_questions.csv`:
```csv
question,company,year,expected_section,gold_answer,answer_type
"New question here",AAPL,2023,Risk Factors,"Expected answer",risk
```

Run: `python -m evaluation.run_evaluation`

---

## 📚 References & Resources

- **SEC EDGAR API**: [sec.gov/developer](https://www.sec.gov/developer)
- **ChromaDB Docs**: [docs.trychroma.com](https://docs.trychroma.com)
- **SentenceTransformers**: [sbert.net](https://www.sbert.net)
- **Groq API**: [console.groq.com](https://console.groq.com)
- **Streamlit**: [docs.streamlit.io](https://docs.streamlit.io)

---

## 🙏 Acknowledgments

- **Anthropic** for Claude API and guidance
- **Groq** for fast LLM inference
- **HuggingFace** for hosting the demo Space
- **ChromaDB** team for the excellent vector database
- **Streamlit** for making ML UIs accessible

---

## 📞 Contact

**Harshita Saraogi** — harshitasaraogi01@gmail.com

**Live Demo**: [https://huggingface.co/spaces/jainharshu/financial-agentic-rag](https://huggingface.co/spaces/jainharshu/financial-agentic-rag)

---
