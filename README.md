<div align="center">

# Financial Agentic RAG — SEC 10-K Question Answering

**An agentic Retrieval-Augmented Generation system that answers natural-language questions over SEC 10-K filings — and a controlled benchmark proving where agentic reasoning beats a vanilla RAG baseline.**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![ChromaDB](https://img.shields.io/badge/Vector_Store-ChromaDB-FF6B6B)](https://www.trychroma.com/)
[![Groq](https://img.shields.io/badge/LLM-Groq_LLaMA_3.1-F55036)](https://groq.com/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![HF Spaces](https://img.shields.io/badge/Live_Demo-HuggingFace_Spaces-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/spaces/jainharshu/financial-agentic-rag)
[![SEC EDGAR](https://img.shields.io/badge/Data-SEC_EDGAR_+_XBRL-1A5276)](https://www.sec.gov/edgar)

[**Live Demo**](https://huggingface.co/spaces/jainharshu/financial-agentic-rag) · [**Source**](https://github.com/jainharshu01/financial-agentic-rag) · [**Author**](https://www.linkedin.com/in/harshita-saraogi/)

</div>

---

## TL;DR

A single-stage dense-retrieval RAG pipeline is the default recipe in most tutorials. On **SEC 10-K filings it fails on the most important question type — extracting exact financial figures.** This project builds an **agentic pipeline** on top of that baseline (query classification → section routing → hybrid dense + lexical retrieval with Reciprocal Rank Fusion → cross-encoder reranking → structured XBRL fact injection → answer validation → retry), then measures both pipelines head-to-head on a **40-question, 80-run benchmark**.

**The headline result:** on questions whose gold answer contains a hard number, the baseline gets **1 of 16** right within a 2% tolerance; the agentic pipeline gets **7 of 16** — a **7× gain** — while also lifting section-routing accuracy, citation grounding, and LLM-judge quality. The cost is latency: roughly **1.4 s → 20.4 s** per question.

---

## Key results

> All numbers below are from the deployed evaluation dashboard (`results_judged.csv`, including LLM-judge scores), produced by the project's automated graders over the full 40-question benchmark run through **both** pipelines (80 runs total).

| Metric | Baseline RAG | Agentic RAG | What it measures |
|---|---:|---:|---|
| **True numeric accuracy** *(±2% vs SEC XBRL)* | **6.3%** (1/16) | **43.8%** (7/16) | Gold-figure questions where the answer contains the correct value within tolerance — not merely *a* number |
| **Section-routing accuracy** | 82% | **90%** | Did retrieval land in the correct 10-K section (Item 1A, Item 7, Item 8, etc.)? |
| **Citation presence** | 67.5% | **92.5%** | Share of answers carrying at least one `[Source N]` citation |
| **Synonym-normalized gold overlap** | 18.2% | **28.0%** | Token recall of gold terms after collapsing financial synonyms (*revenue = net sales = total revenues*) |
| **LLM-judge score** *(risk + descriptive, 0–1)* | 0.30 | **0.43** | Second-opinion quality on prose answers (`llama-3.3-70b`), directional only |
| **Avg retrieval similarity** | 75.0% | **79.6%** | Mean cosine similarity of retrieved chunks |
| **Top retrieval similarity** | 76.5% | **100%** | Similarity of the single best retrieved chunk |
| **Comparison completeness** | 100% | 100% | Both fiscal years cited on multi-year comparative questions |
| **Avg latency** | **1.40 s** | 20.40 s | Wall-clock per question |
| **Retry rate** | — | 10% | Share of agentic answers that triggered a validation-driven retry |

### Section-routing accuracy by question type

| Question type | Baseline | Agentic |
|---|---:|---:|
| Comparative | 100% | 100% |
| Numeric | 90% | 90% |
| Risk | 90% | **100%** |
| Descriptive | 58% | **75%** |

**How to read this.** The agentic pipeline's gains are concentrated exactly where structured grounding and routing help — **numeric extraction, risk routing, and descriptive coverage.** On comparative questions both pipelines already route well; on prose-only risk/descriptive answers the bottleneck is generation, not retrieval, so the gap narrows. This is the expected and honest shape of the result, not a uniform "agentic wins everything."

### A representative case — *"What was Apple's total net sales in its FY2024 10-K?"*

> **Gold:** Total net sales were **$391.0 billion** ($391,035 M) for fiscal 2024.

- **Baseline** *(10.97 s)* — retrieves table fragments but hedges across several paragraphs and never commits to a figure, concluding the documents *"do not"* contain the total.
- **Agentic** *(24.78 s)* — **`$391.04 billion (Source 1)`** — resolved exactly via the SEC XBRL fact path injected as the authoritative `Source 1`.

This single example is the whole thesis in miniature: the same generator, given a structured exact fact and reranked evidence, stops hallucinating uncertainty and answers correctly.

---

## What's actually engineered here

Most public RAG demos are *embedder + Chroma + LLM*. On financial filings that minimal recipe fails on numeric facts, mis-routes risk questions, and drops citations. This build adds **five concrete layers**, each motivated by an observed failure mode:

1. **Table-aware parsing.** `pandas.read_html` extracts every `<table>` and serializes it to a Markdown pipe table **before** prose is flattened, so `Net sales | 391,035` survives as a row instead of collapsing into orphaned vertical numbers. Tables are also written to a sidecar `*.tables.json` for dedicated table chunks.
2. **Order-constrained, context-injected chunking.** A 10-K is full of a table-of-contents and in-prose cross-references (*"see Item 1A"*, *"Item 7 of Part II"*) that naïvely look like section headings. The chunker classifies every candidate heading against five rejection rules and reconstructs the canonical **Item spine in order**, so *Risk Factors* and *MD&A* no longer collapse to a single mislabeled mega-chunk. Each chunk is prefixed with a context header — `[Apple Inc. (AAPL) | FY2024 | 10-K | Section: Risk Factors]` — giving the embedder global context the metadata alone never reached.
3. **Hybrid retrieval with Reciprocal Rank Fusion.** A `rank_bm25` lexical index runs alongside dense retrieval (`BAAI/bge-small-en-v1.5`), because exact tokens — `diluted EPS`, `FY2024`, `$391,035` — are what finance queries hinge on and embeddings blur them. The two ranked lists are fused with **RRF (k=60)**, which operates on ranks rather than incompatible score scales, then reranked by a **cross-encoder** (`ms-marco-MiniLM-L-6-v2`) that scores each (query, chunk) pair jointly.
4. **SEC XBRL exact-fact path.** Numeric queries pull GAAP-tagged values straight from the free SEC **XBRL `companyfacts` API** (cached once at ingestion, never called at query time) and inject them as an authoritative **Source 1** ahead of the LLM. Gross margin is supported as a derived ratio (`GrossProfit / Revenue`) per fiscal year. This eliminates an entire class of numeric hallucination.
5. **Query-aware routing, validation, and retry.** A rule-based classifier tags each query (numeric / comparative / risk / descriptive); a router maps it to section-specific retrieval strategies (including dedicated **cross-company** and **multi-year comparative** paths); the generated answer is validated for citations, year coverage, numeric presence, and unsupported-claim phrasing, with a **single targeted retry** on failure.

---

## System architecture

```
                                User question
                                       │
                          ┌────────────┴────────────┐
                          │   Query Classifier      │  rule-based
                          │ numeric / risk / desc / │
                          │       comparative       │
                          └────────────┬────────────┘
                                       │
                          ┌────────────┴────────────┐
                          │     Section Router      │  Item 1A / 7 / 8 / 1
                          │ + cross-company &       │
                          │   multi-year paths      │
                          └────────────┬────────────┘
                                       │
            ┌──────────────────────────┼──────────────────────────┐
            │                          │                          │
   ┌────────▼────────┐    ┌────────────▼────────────┐    ┌────────▼────────┐
   │ ChromaDB        │    │   BM25 lexical index    │    │ SEC XBRL facts  │
   │ bge-small-en,   │    │ rank_bm25 + financial   │    │ companyfacts    │
   │ context-        │    │ synonym expansion       │    │ API (exact      │
   │ injected chunks │    │ (query side)            │    │ GAAP numbers)   │
   └────────┬────────┘    └────────────┬────────────┘    └────────┬────────┘
            │                          │                          │
            └──────────────┬───────────┘                          │
                           │                                      │
                  ┌────────▼─────────┐                            │
                  │ Reciprocal Rank  │                            │
                  │ Fusion  (k=60)   │                            │
                  └────────┬─────────┘                            │
                           │                                      │
                  ┌────────▼─────────┐                            │
                  │ Cross-encoder    │                            │
                  │ rerank (top-k)   │                            │
                  └────────┬─────────┘                            │
                           │                                      │
                           └──────────────┬───────────────────────┘
                                          │  XBRL injected as Source 1
                              ┌───────────▼───────────┐
                              │  Groq LLM             │
                              │llama-3.3-70b-versatile│
                              │  (temperature 0.2)    │
                              └───────────┬───────────┘
                                          │
                              ┌───────────▼───────────┐
                              │ Answer validator      │
                              │ citations / years /   │
                              │ numeric / grounded    │
                              └───────────┬───────────┘
                                          │
                                  (targeted retry if invalid)
                                          │
                                  ┌───────▼────────┐
                                  │ Final answer   │
                                  │ + [Source N]   │
                                  └────────────────┘
```

The **baseline** pipeline is the dashed inner path only: dense retrieval → LLM → answer. Everything else is the agentic delta under test.

---

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Embeddings | `BAAI/bge-small-en-v1.5` | 384-dim, 512-token context; retrieval-instruction prefix applied query-side |
| Vector store | ChromaDB (persistent) | Zero-config, metadata filtering, fits HF Spaces |
| Lexical retrieval | `rank_bm25` (in-memory `BM25Okapi`) | Exact-token matching is decisive on financial text |
| Fusion | Reciprocal Rank Fusion (k=60) | Rank-based, robust to score-scale mismatch |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | ~80 MB; true joint (query, doc) scoring |
| Generation LLM | `llama-3.3-70b-versatile` via Groq | Fast, free-tier; **identical generator across both pipelines** for a fair comparison |
| LLM judge | `llama-3.3-70b-versatile` via Groq | Stronger second-opinion scorer, run separately |
| Structured numerics | SEC XBRL `companyfacts` API | Free, no key, exact GAAP-tagged values |
| Parsing | BeautifulSoup + `pandas.read_html` | Table-aware HTML → Markdown |
| UI / Dashboard | Streamlit + Plotly | Single-file deploy, identical local ↔ HF Spaces |

---

## Dataset

**10 annual 10-K filings** pulled from SEC EDGAR: **Apple, Microsoft, Amazon, Tesla, Alphabet**, fiscal years **2023 and 2024**.

> Note: filing year ≠ fiscal year for December year-end companies (Amazon, Tesla, Alphabet file their FY2024 10-K in early 2025). Fiscal years are resolved from XBRL period end-dates, not accession numbers.

---

## Repository layout

```
financial-agentic-rag/
├── app.py                              # Streamlit chat UI
├── requirements.txt · runtime.txt      # python-3.11
│
├── src/
│   ├── ingestion/
│   │   ├── download_filings.py         # SEC EDGAR HTML pull
│   │   └── xbrl_facts.py               # XBRL companyfacts → exact GAAP numbers + gross_margin
│   ├── preprocessing/
│   │   ├── parse_filings.py            # Table-aware HTML → Markdown + tables.json
│   │   ├── chunk_filings.py            # Order-constrained, context-injected chunking
│   │   └── verify_parsing.py
│   ├── retrieval/
│   │   ├── build_vectorstore.py        # bge-small embeddings → ChromaDB
│   │   ├── bm25_index.py               # In-memory BM25 + synonym expansion
│   │   ├── fusion.py                   # Reciprocal Rank Fusion (k=60)
│   │   ├── reranker.py                 # cross-encoder rerank
│   │   └── baseline_rag.py             # Static dense-only pipeline
│   ├── agent/
│   │   ├── query_classifier.py · router.py
│   │   ├── query_parser.py · company_parser.py · financial_synonyms.py
│   │   ├── self_check.py · answer_validator.py
│   │   └── agentic_rag.py              # Main agentic pipeline
│   └── utils/sec_urls.py
│
└── evaluation/
    ├── evaluation_questions.csv        # 40-question benchmark
    ├── run_evaluation.py               # Runs every Q through both pipelines
    ├── metrics.py · metrics_synonym_patch.py
    ├── numeric_grader.py               # Objective ±2% numeric grading
    ├── llm_judge.py                    # LLM-as-judge (70B, run separately)
    ├── dashboard.py                    # Streamlit results dashboard
    └── results.csv
```

---


## Evaluation methodology

The benchmark is **40 manually-curated questions** — 12 descriptive, 10 numeric, 10 risk, 8 comparative — each run through both pipelines. There is no single metric that fairly grades both *"$391 billion"* and *"Apple faces global-competition and supply-chain risks…"*, so question types are graded differently:

- **Numeric / comparative answers with a gold figure** are graded **objectively** by `numeric_grader.py`: dollar/percent values are extracted from gold and generated answers, multipliers normalized (so *391 billion* ≡ *391,035 million*), and an answer is correct only if **every** gold figure is matched within **2% relative tolerance**. No human judgment.
- **Risk / descriptive prose answers** are graded by **synonym-normalized token recall** (`metrics_synonym_patch.py`): financial synonyms are collapsed before computing overlap, so SEC-correct terminology is not penalized.
- **LLM-as-judge** (`llm_judge.py`, `llama-3.3-70b`) provides a second-opinion score on prose answers, reported as **directional cross-validation, not ground truth.**

### Honest limitations

- **40 questions is small** — a ±5% metric move can be one or two questions flipping. Treat the *direction and magnitude* of gains as the result, not the third significant figure.
- **LLM-judge bias** — a model scoring another model tends to reward longer answers; reported as secondary only.
- **Segment financials** (AWS revenue, Tesla automotive revenue, Google advertising revenue) are not standard top-line GAAP tags; the XBRL path can't resolve them and falls back to retrieved tables — the main remaining numeric gap.
- **Latency** — the cross-encoder rerank dominates agentic latency. Setting `USE_RERANK = False` trades quality for speed.
- The deployed dashboard reads `results_judged.csv`; an intermediate `results.csv` from a partial run may show different figures.

---

## Future work

- SEC XBRL **`frames` API** for segment-tagged concepts to close the segment-revenue gap.
- A **finance-tuned embedding model** (e.g. Fin-E5) to lift dense retrieval.
- Conversational memory across turns via `st.session_state`.
- Expanding the benchmark beyond 40 questions for tighter confidence intervals.

---

## Author

**Harshita Saraogi** — M.Sc. Data Science & Analytics (2024–2026), Maulana Abul Kalam Azad University of Technology (MAKAUT)
Major Project · Faculty Guide: Prof. Ashutosh Kar

- 📧 harshitasaraogi01@gmail.com
- 🔗 [GitHub](https://github.com/jainharshu01/financial-agentic-rag) · [Live Demo](https://huggingface.co/spaces/jainharshu/financial-agentic-rag)

## Acknowledgments

SEC EDGAR (filings + XBRL companyfacts API) · Groq (free-tier inference) · ChromaDB, sentence-transformers, rank_bm25, Streamlit maintainers.