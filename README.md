# Financial Agentic RAG — SEC 10-K Question Answering

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)

A research-grade Retrieval-Augmented Generation (RAG) system for answering natural-language questions over SEC 10-K filings, comparing a **baseline static pipeline** against an **agentic pipeline** that adds query classification, hybrid retrieval (dense + BM25 with Reciprocal Rank Fusion), cross-encoder reranking, exact-fact lookup via SEC XBRL, answer validation, and retry logic.

Built on 20 annual 10-K filings from 5 large US technology companies (Apple, Amazon, Alphabet, Microsoft, Tesla; fiscal years 2022–2025).

---

## Research question

> *Can an agentic RAG pipeline — combining hybrid retrieval, structured XBRL fact injection, cross-encoder reranking, and answer validation — produce measurably more accurate and better-grounded answers on SEC 10-K filings than a single-stage dense-retrieval baseline, on a benchmark of 40 manually-curated questions across numeric, comparative, risk, and descriptive query types?*

---

## Headline results (40-question benchmark, 80 runs)

| Metric | Baseline RAG | Agentic RAG |
|---|---:|---:|
| **Numeric accuracy** *(exact, vs SEC XBRL, ±2% tol.)* | **0 / 16** (0%) | **10 / 16** (62.5%) |
| Section accuracy | 75% | 75% |
| Citation rate | 90% | 88% |
| Comparison completeness | 100% | 87.5% |
| Avg retrieval similarity | 65.4% | 76.9% |
| Avg latency | 1.4 s | 20.6 s |
| Retry frequency | n/a | 12.5% |

**Headline finding.** On objective numeric questions, the baseline failed every single one (typically returning *"Insufficient evidence in the provided documents"*); the agentic pipeline answered 62.5% correctly to within 2% of the XBRL ground truth. The cost is a ~15× latency increase, driven primarily by the cross-encoder reranker and additional generation passes.

The two pipelines are roughly equivalent on prose-style risk and descriptive questions, where the bottleneck is generation rather than retrieval. The agentic pipeline's wins are concentrated where structured grounding helps — numeric and comparative queries.

---

## What's novel about this build

Most public RAG tutorials wire up a single embedder + Chroma + an LLM. This project is built specifically for SEC filings, where that minimal recipe fails on the most common question type (numeric facts), and it adds five concrete layers on top:

1. **Table-aware parsing.** HTML tables are extracted with `pandas.read_html` and serialized as Markdown *before* the prose is flattened, so row/column relationships survive into the chunks.
2. **Context-injected chunking.** Every chunk's embedded text begins with a header (`[Apple Inc. (AAPL) | FY2024 | 10-K | Section: Risk Factors]`), giving the embedder global context the metadata alone never reached.
3. **Hybrid retrieval with Reciprocal Rank Fusion.** A BM25 index runs alongside dense retrieval (`bge-small-en-v1.5`); results are fused by rank using RRF (k=60), then reranked with a cross-encoder (`ms-marco-MiniLM-L-6-v2`).
4. **XBRL exact-fact path.** Numeric queries pull exact values from the free SEC XBRL companyfacts API (`data.sec.gov`) and inject them as an authoritative source ahead of the LLM call — eliminating an entire class of numeric hallucination.
5. **Query-aware routing and validation.** A classifier routes queries by type (numeric / risk / descriptive / comparative) to type-specific retrieval strategies; generated answers are validated for citations, year coverage, and numeric presence, with a retry on failure.

---

## System architecture

```
                                User question
                                       │
                          ┌────────────┴────────────┐
                          │   Query Classifier      │ classifier
                          │ numeric / risk / desc / │
                          │       comparative       │
                          └────────────┬────────────┘
                                       │
                          ┌────────────┴────────────┐
                          │     Section Router      │  router
                          │ Risk Factors / MD&A /   │
                          │ Business / FinStmts ... │
                          └────────────┬────────────┘
                                       │
            ┌──────────────────────────┼──────────────────────────┐
            │                          │                          │
   ┌────────▼────────┐    ┌────────────▼────────────┐    ┌────────▼────────┐
   │ ChromaDB        │    │   BM25 lexical index    │    │ SEC XBRL facts  │
   │ (bge-small,     │    │ (rank_bm25 + financial  │    │ (companyfacts   │
   │  context-       │    │  synonym expansion)     │    │  API, exact     │
   │  injected       │    └────────────┬────────────┘    │  GAAP numbers)  │
   │  chunks)        │                 │                 └────────┬────────┘
   └────────┬────────┘                 │                          │
            │                          │                          │
            └──────────────┬───────────┘                          │
                           │                                      │
                  ┌────────▼─────────┐                            │
                  │ Reciprocal Rank  │                            │
                  │ Fusion (k=60)    │                            │
                  └────────┬─────────┘                            │
                           │                                      │
                  ┌────────▼─────────┐                            │
                  │ Cross-encoder    │                            │
                  │ rerank           │                            │
                  └────────┬─────────┘                            │
                           │                                      │
                           └──────────────┬───────────────────────┘
                                          │
                              ┌───────────▼───────────┐
                              │  Groq LLM             │
                              │  llama-3.3-70b        │
                              │  (XBRL as Source 1)   │
                              └───────────┬───────────┘
                                          │
                              ┌───────────▼───────────┐
                              │ Answer validator      │
                              │ (citations / years /  │
                              │  numeric / grounded)  │
                              └───────────┬───────────┘
                                          │
                                  (retry if invalid)
                                          │
                                  ┌───────▼────────┐
                                  │ Final answer   │
                                  │ + citations    │
                                  └────────────────┘
```

---

## Repository layout

```
financial-agentic-rag/
├── app.py                              # Streamlit chat UI
├── requirements.txt
├── runtime.txt                         # python-3.11
├── README.md                           # this file
├── .env.example                        # GROQ_API_KEY + SEC_USER_AGENT
│
├── src/
│   ├── ingestion/
│   │   ├── download_filings.py         # SEC EDGAR HTML pull (5 tickers, last 3 yrs)
│   │   └── xbrl_facts.py               # SEC XBRL companyfacts API → exact GAAP numbers
│   ├── preprocessing/
│   │   ├── parse_filings.py            # Table-aware HTML → Markdown text + tables.json
│   │   ├── chunk_filings.py            # Token-aware, context-injected chunking
│   │   └── verify_parsing.py
│   ├── retrieval/
│   │   ├── build_vectorstore.py        # bge-small embeddings → ChromaDB
│   │   ├── bm25_index.py               # In-memory BM25 + synonym expansion
│   │   ├── fusion.py                   # Reciprocal Rank Fusion
│   │   ├── reranker.py                 # cross-encoder/ms-marco-MiniLM-L-6-v2
│   │   ├── baseline_rag.py             # Static dense-only pipeline
│   │   └── rag_pipeline.py             # (legacy; safe to delete)
│   ├── agent/
│   │   ├── query_classifier.py
│   │   ├── router.py
│   │   ├── query_parser.py             # year extraction
│   │   ├── company_parser.py           # ticker detection from natural language
│   │   ├── financial_synonyms.py       # revenue ↔ net sales, etc.
│   │   ├── self_check.py               # distance-based retry trigger
│   │   ├── answer_validator.py         # citations / year coverage / numeric checks
│   │   └── agentic_rag.py              # main agentic pipeline
│   └── utils/
│       └── sec_urls.py                 # Build SEC EDGAR filing URLs
│
├── evaluation/
│   ├── evaluation_questions.csv        # 40 manually-curated Q&A pairs
│   ├── run_evaluation.py               # Runs every Q through both pipelines
│   ├── metrics.py                      # Section accuracy, citations, latency, etc.
│   ├── metrics_synonym_patch.py        # Synonym-normalized gold-overlap metric
│   ├── numeric_grader.py               # Objective numeric grading vs gold
│   ├── llm_judge.py                    # LLM-as-judge for prose answers (optional)
│   ├── dashboard.py                    # Streamlit results dashboard
│   └── results.csv                     # Auto-generated by run_evaluation
│
└── data/
    ├── raw_filings/                    # SEC EDGAR HTML (downloaded; gitignored)
    ├── processed/                      # Parsed text + .tables.json (gitignored)
    ├── chunks/                         # JSON chunks (gitignored)
    ├── vectorstore/                    # ChromaDB persistent store (gitignored)
    └── xbrl/                           # XBRL facts cache + facts_lookup.json
```

---

## Quick start

### Prerequisites

- Python 3.11
- A free [Groq API key](https://console.groq.com) (`GROQ_API_KEY`)
- A descriptive User-Agent for the SEC API (`SEC_USER_AGENT` — e.g. `"YourProject your-email@example.com"`)

### Setup

```bash
git clone <repo-url>
cd financial-agentic-rag
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env       # then edit .env with your keys
```

### Build the data pipeline (once)

```bash
# 1. Download 10-K HTML filings from EDGAR
python -m src.ingestion.download_filings

# 2. Parse HTML → text (with Markdown tables preserved)
python -m src.preprocessing.parse_filings

# 3. Fetch exact financial numbers from SEC XBRL
python -m src.ingestion.xbrl_facts

# 4. Chunk filings with context headers
python -m src.preprocessing.chunk_filings

# 5. Build the ChromaDB vector store
python -m src.retrieval.build_vectorstore
```

### Run the app

```bash
streamlit run app.py
```

### Run the benchmark

```bash
python -m evaluation.run_evaluation       # ~20 min on free Groq tier
python -m evaluation.numeric_grader        # Objective numeric scoring
python -m evaluation.metrics_synonym_patch # Token overlap on prose answers
streamlit run evaluation/dashboard.py
```

---

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Embeddings | `BAAI/bge-small-en-v1.5` | 384-dim, 512-token context, strong on retrieval benchmarks |
| Vector store | ChromaDB (persistent) | Zero-config, metadata-filtering, fits HF Spaces |
| Lexical retrieval | `rank_bm25` (in-memory) | Pure-Python, exact-token matching critical for finance |
| Fusion | Reciprocal Rank Fusion (k=60) | Rank-based, robust to score-scale mismatch |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | ~80MB, real query-document scoring (not bi-encoder) |
| LLM | `llama-3.3-70b-versatile` via Groq | Fast inference, free tier sufficient for benchmark |
| Structured numerics | SEC XBRL companyfacts API | Free, no key, exact GAAP-tagged values |
| UI | Streamlit | Single-file deployable; same on local + HF Spaces |

---

## Evaluation methodology

The benchmark contains **40 manually-curated questions** spanning four types:

| Type | Count | Example |
|---|---:|---|
| numeric | 10 | "What was Apple's total net revenue in fiscal year 2024?" |
| comparative | 8 | "Compare Tesla's total revenue between 2023 and 2024" |
| risk | 10 | "What cybersecurity risks did Microsoft disclose in 2024?" |
| descriptive | 12 | "What are Apple's main business segments?" |

### Grading approach

Different question types are graded differently, because there is no single metric that works for both "$391 billion" and "Apple faces risks including global competition, supply chain concentration ...":

- **Numeric & comparative answers with gold figures (22 questions)** — graded objectively by `numeric_grader.py`. The grader extracts dollar/percent values from gold and generated answers and counts an answer correct if every gold figure is matched to within 2% (absorbing rounding like "391 billion" ≈ "391.04 billion"). No human judgment.
- **Risk & descriptive answers (18 questions)** — graded by synonym-normalized token overlap (`metrics_synonym_patch.py`). Financial synonyms ("revenue" ↔ "net sales", "profit" ↔ "net income") are collapsed before computing recall, so SEC-correct terminology is not penalized.
- **Optional LLM-as-judge** (`llm_judge.py`) — a second-opinion score on the 18 prose answers using the same Groq LLM with a strict rubric. Useful as cross-validation; **not** treated as ground truth.

### Honest limitations of the methodology

- 40 questions is a small benchmark; ±5% movement on a metric can be one or two questions flipping.
- The LLM-as-judge is a model judging another model's output and tends to reward longer, more detailed answers; treat its scores as directional, not authoritative.
- Three numeric questions (AWS revenue, Tesla automotive revenue, Google advertising revenue) target *segment* figures that are not standard top-line GAAP tags. The XBRL path cannot resolve these and falls back to retrieved tables.
- Gross margin is computed as a derived ratio (`GrossProfit / Revenue`) per fiscal year; not all filings tag both consistently.

---

## Known limitations and future work

- **Latency.** The cross-encoder rerank dominates agentic latency (~20s per question on free-tier infra). On HuggingFace Spaces' free CPU this can be higher. Setting `USE_RERANK = False` in `src/agent/agentic_rag.py` skips it, falling back to RRF only — faster but with some quality loss.
- **Segment financials.** Questions targeting business segments (e.g. "Amazon's AWS revenue") need either the SEC XBRL `frames` API for segment-tagged concepts, or richer table-chunk routing. Currently underperforms.
- **LLM-as-judge bias.** Reported as a secondary metric only; spot-checking 5 random rows is recommended.
- **Conversational memory.** Each question is stateless. Could be added with `st.session_state` if needed.
- **No re-fine-tuning.** A finance-specific embedding model (e.g. Fin-E5) would likely lift retrieval further but is out of scope for this iteration.

---

## Acknowledgments

- **SEC EDGAR** for free, machine-readable filings and the XBRL companyfacts API.
- **Anthropic** for design discussion and architecture review during development.
- **Groq** for free-tier LLM inference.
- **ChromaDB**, **sentence-transformers**, **rank_bm25**, and **Streamlit** maintainers.

---

## Contact

**Harshita Saraogi** — harshitasaraogi01@gmail.com

**Live Demo**: [https://huggingface.co/spaces/jainharshu/financial-agentic-rag](https://huggingface.co/spaces/jainharshu/financial-agentic-rag)

---
