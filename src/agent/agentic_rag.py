"""
src/agent/agentic_rag.py   (QUALITY-FIRST — reverts the latency change)

WHAT CHANGED vs the latency-optimized version
---------------------------------------------
The latency-optimized version broke quality:
  - candidate_k 20 -> 10 on EVERY call
  - balanced paths (cross-company, multi-year) skipped per-call reranking
Together these collapsed the candidate pool that the cross-encoder picks
from, so section accuracy crashed from 75% to 18% and citations from 88%
to 22%. This version reverts to the proven quality settings.

KEPT FROM THE LATENCY VERSION (safe, non-quality changes):
  - gross_margin routing in NUMERIC_CONCEPT_KEYWORDS
  - XBRL-only answers report avg_similarity = 0.0 instead of fake 100,
    so dashboard averages aren't inflated

ALL OTHER BEHAVIOR matches the version that produced 88% citations /
75% section accuracy / 62% numeric correct.
"""

import os
import chromadb

from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

from src.agent.query_classifier import classify_query
from src.agent.router import build_retrieval_strategy
from src.agent.query_parser import extract_years
from src.agent.company_parser import extract_companies, is_cross_company_query
from src.agent.self_check import should_retry
from src.agent.answer_validator import validate_answer

from src.retrieval.bm25_index import get_bm25_index
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.reranker import rerank_results
from src.ingestion.xbrl_facts import get_fact, format_value


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Quality-first: 20 candidates per channel, rerank EVERY call.
CANDIDATE_K = 20
USE_RERANK = True

embedding_model = SentenceTransformer(EMBED_MODEL_NAME)
client = chromadb.PersistentClient(path="data/vectorstore")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Note: gross_margin must come BEFORE gross_profit so "gross margin"
# queries route to the derived ratio, not gross_profit dollars.
NUMERIC_CONCEPT_KEYWORDS = {
    "gross_margin": ["gross margin", "gross profit margin"],
    "gross_profit": ["gross profit"],
    "revenue": ["revenue", "net sales", "sales", "total revenue"],
    "net_income": ["net income", "profit", "net earnings", "earnings"],
    "operating_income": ["operating income", "operating profit"],
    "total_assets": ["total assets", "assets"],
    "total_liabilities": ["total liabilities", "liabilities"],
    "diluted_eps": ["eps", "earnings per share", "diluted eps"],
    "rnd_expense": ["research and development", "r&d", "rnd"],
    "operating_cash_flow": ["operating cash flow", "cash from operations"],
}


def _embed_query(question):
    return embedding_model.encode(
        QUERY_PREFIX + question, normalize_embeddings=True
    ).tolist()


# ============================================================
# FILTER HELPERS
# ============================================================

def build_where_filter(filters):
    if len(filters) == 0:
        return None
    if len(filters) == 1:
        return filters[0]
    return {"$and": filters}


def resolve_companies(question: str, user_company) -> list:
    query_companies = extract_companies(question)
    if query_companies:
        return query_companies
    if user_company:
        return [user_company]
    return []


# ============================================================
# HYBRID RETRIEVE — dense + BM25 -> RRF -> cross-encoder rerank
# ============================================================

def hybrid_retrieve(collection, question, where, top_k,
                    rerank=True, candidate_k=CANDIDATE_K):
    """Unified hybrid retrieval, always returning a Chroma-style dict."""
    query_embedding = _embed_query(question)

    dense = collection.query(
        query_embeddings=[query_embedding],
        n_results=candidate_k,
        where=where,
    )

    bm25 = get_bm25_index()
    sparse = bm25.query(question, top_k=candidate_k, where=where)

    fused = reciprocal_rank_fusion(
        dense, sparse, top_k=(candidate_k if rerank else top_k)
    )

    if rerank and USE_RERANK and fused["ids"][0]:
        return rerank_results(question, fused, top_k=top_k)
    return fused


# ============================================================
# XBRL NUMERIC FAST-PATH
# ============================================================

def detect_concept(question: str):
    q = question.lower()
    for concept, kws in NUMERIC_CONCEPT_KEYWORDS.items():
        if any(kw in q for kw in kws):
            return concept
    return None


def get_xbrl_facts(question, companies, years):
    concept = detect_concept(question)
    if not concept or not companies:
        return []
    target_years = years if years else _recent_years()
    facts = []
    for company in companies:
        for yr in target_years:
            val = get_fact(company, concept, yr)
            if val is not None:
                facts.append(
                    (company, yr, concept, format_value(concept, val))
                )
    return facts


def _recent_years():
    return [2024, 2023, 2022]


# ============================================================
# RETRIEVE USING DYNAMIC STRATEGY
# ============================================================

def retrieve_chunks_agentic(question, company=None, year=None):
    strategy = build_retrieval_strategy(question)
    query_type = strategy["query_type"]
    top_k = strategy["top_k"]
    section = strategy["section"]
    sections_allowed = strategy.get("sections_allowed")

    print("\nRetrieval Strategy:")
    print(strategy)

    collection = client.get_collection(name="section_chunks")

    query_years = extract_years(question)
    target_companies = resolve_companies(question, company)
    cross_company = len(target_companies) >= 2

    print(f"Target companies: {target_companies or '[all]'}")
    print(f"Cross-company query: {cross_company}")

    # -----------------------------------------------------------
    # CROSS-COMPANY BALANCED — hybrid per company, rerank per call
    # -----------------------------------------------------------
    if cross_company:
        chunks_per_company = max(2, top_k // len(target_companies) + 1)
        all_ids, all_docs, all_metas, all_dists = [], [], [], []

        for ticker in target_companies:
            cf = [{"company": ticker}]
            if len(query_years) == 1:
                cf.append({"year": query_years[0]})
            elif len(query_years) > 1:
                cf.append({"$or": [{"year": y} for y in query_years]})
            elif year:
                cf.append({"year": year})
            if section:
                cf.append({"section": section})
            elif sections_allowed:
                cf.append({"$or": [{"section": s} for s in sections_allowed]})

            where = build_where_filter(cf)
            r = hybrid_retrieve(collection, question, where,
                                top_k=chunks_per_company, rerank=True)
            if r["ids"][0]:
                all_ids += r["ids"][0]
                all_docs += r["documents"][0]
                all_metas += r["metadatas"][0]
                all_dists += r["distances"][0]
                print(f"  {ticker}: {len(r['ids'][0])} chunks")

        results = {"ids": [all_ids], "documents": [all_docs],
                   "metadatas": [all_metas], "distances": [all_dists]}
        return results, strategy

    # -----------------------------------------------------------
    # MULTI-YEAR COMPARATIVE — hybrid per year, rerank per call
    # -----------------------------------------------------------
    if query_type == "comparative" and len(query_years) >= 2:
        print(f"Multi-year comparative. Years: {query_years}")
        chunks_per_year = 3
        single_company = target_companies[0] if target_companies else None
        all_ids, all_docs, all_metas, all_dists = [], [], [], []

        for yr in query_years:
            yf = [{"year": yr}]
            if single_company:
                yf.append({"company": single_company})
            if section:
                yf.append({"section": section})
            elif sections_allowed:
                yf.append({"$or": [{"section": s} for s in sections_allowed]})

            where = build_where_filter(yf)
            r = hybrid_retrieve(collection, question, where,
                                top_k=chunks_per_year, rerank=True)
            if r["ids"][0]:
                all_ids += r["ids"][0]
                all_docs += r["documents"][0]
                all_metas += r["metadatas"][0]
                all_dists += r["distances"][0]
                print(f"  Year {yr}: {len(r['ids'][0])} chunks")

        results = {"ids": [all_ids], "documents": [all_docs],
                   "metadatas": [all_metas], "distances": [all_dists]}
        return results, strategy

    # -----------------------------------------------------------
    # STANDARD — single hybrid call, reranked
    # -----------------------------------------------------------
    filters = []
    if target_companies:
        if len(target_companies) == 1:
            filters.append({"company": target_companies[0]})
        else:
            filters.append({"$or": [{"company": c} for c in target_companies]})
    if len(query_years) == 1:
        filters.append({"year": query_years[0]})
    elif len(query_years) > 1:
        filters.append({"$or": [{"year": y} for y in query_years]})
    elif year:
        filters.append({"year": year})
    if section:
        filters.append({"section": section})
    elif sections_allowed:
        filters.append({"$or": [{"section": s} for s in sections_allowed]})

    where = build_where_filter(filters)
    results = hybrid_retrieve(collection, question, where, top_k=top_k)
    return results, strategy


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(question, results, xbrl_facts=None):
    context_parts = []

    if xbrl_facts:
        lines = [
            f"{c} FY{y} {concept.replace('_', ' ')}: {val}"
            for (c, y, concept, val) in xbrl_facts
        ]
        context_parts.append(
            "[Source 1] AUTHORITATIVE XBRL FINANCIAL FACTS "
            "(exact, from SEC structured data)\n\n" + "\n".join(lines)
        )

    offset = len(context_parts)
    for i in range(len(results["ids"][0])):
        doc = results["documents"][0][i][:600]
        meta = results["metadatas"][0][i]
        context_parts.append(
            f"[Source {offset + i + 1}] Company: {meta['company']} | "
            f"Year: {meta['year']} | Section: {meta['section']} \n\n{doc}"
        )

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are a financial document analyst.

Answer the user's question using ONLY the provided SEC filing context.

RULES:
1. Do NOT use outside knowledge.
2. Cite sources using (Source X) format — always include the parentheses.
3. If AUTHORITATIVE XBRL FINANCIAL FACTS are provided, prefer those exact
   numbers over any figure parsed from narrative text.
4. For cross-company comparisons, explicitly mention EACH company and cite sources from each.
5. For multi-year comparisons, explicitly mention BOTH years and cite sources from each.
6. Include specific numbers when relevant.
7. If evidence is insufficient, say: "Insufficient evidence in the provided documents."

Context:
{context}

Question:
{question}

Answer:"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content


# ============================================================
# AGENTIC PIPELINE
# ============================================================

def agentic_answer(question, company=None, year=None):
    print("\n" + "=" * 60)
    print("AGENTIC RAG PIPELINE")
    print("=" * 60)
    print(f"\nQuestion: {question}")

    results, strategy = retrieve_chunks_agentic(
        question=question, company=company, year=year
    )

    xbrl_facts = []
    if strategy["query_type"] in ("numeric", "comparative"):
        companies = resolve_companies(question, company)
        years = extract_years(question)
        xbrl_facts = get_xbrl_facts(question, companies, years)
        if xbrl_facts:
            print(f"\nXBRL exact facts injected: {len(xbrl_facts)}")
            for f in xbrl_facts:
                print(f"  {f[0]} FY{f[1]} {f[2]} = {f[3]}")

    distances = results["distances"][0]
    if not distances and not xbrl_facts:
        return {
            "answer": "No relevant documents found.",
            "results": results, "strategy": strategy, "retry_used": False,
            "validation": None, "avg_similarity": 0, "top_similarity": 0,
            "retrieved_chunks": 0,
        }

    if distances:
        avg_distance = round(sum(distances) / len(distances), 4)
        avg_similarity = round((1 - avg_distance) * 100, 2)
        top_similarity = round((1 - min(distances)) * 100, 2)
    else:
        # XBRL-only answer: report 0 similarity (no chunks retrieved).
        avg_similarity = top_similarity = 0.0

    print("\nRetrieved Sources:\n")
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        dist = results["distances"][0][i]
        print(f"[{i+1}] {meta['company']} | Year: {meta['year']} | "
              f"{meta['section']} | Distance: {dist:.4f}")

    print("\nGenerating answer...\n")
    answer = generate_answer(question, results, xbrl_facts=xbrl_facts)
    print("\nInitial Answer:\n")
    print(answer)

    query_years = extract_years(question)
    validation = validate_answer(
        answer=answer, query=question,
        query_type=strategy["query_type"], query_years=query_years,
    )
    print("\nValidation Result:")
    print(validation)

    # Retry logic: distance-based check + validation-failure check.
    retry = should_retry(results, answer) if distances else False
    if not retry and not validation["overall_valid"]:
        retry = True
        print("Retry triggered by validation failure.")

    if retry:
        print("\nRetrying with broader hybrid retrieval...\n")
        collection = client.get_collection(name="section_chunks")

        retry_filters = []
        target_companies = resolve_companies(question, company)
        if target_companies:
            if len(target_companies) == 1:
                retry_filters.append({"company": target_companies[0]})
            else:
                retry_filters.append(
                    {"$or": [{"company": c} for c in target_companies]})
        sec = strategy.get("section")
        sections_allowed = strategy.get("sections_allowed")
        if sec:
            retry_filters.append({"section": sec})
        elif sections_allowed:
            retry_filters.append(
                {"$or": [{"section": s} for s in sections_allowed]})

        retry_where = build_where_filter(retry_filters)
        retry_results = hybrid_retrieve(
            collection, question, retry_where, top_k=8
        )

        print("\nRetry Retrieved Sources:\n")
        for i in range(len(retry_results["ids"][0])):
            meta = retry_results["metadatas"][0][i]
            dist = retry_results["distances"][0][i]
            print(f"[{i+1}] {meta['company']} | Year: {meta['year']} | "
                  f"{meta['section']} | Distance: {dist:.4f}")

        print("\nGenerating Retry Answer...\n")
        retry_answer = generate_answer(
            question, retry_results, xbrl_facts=xbrl_facts
        )
        print("\nRetry Answer:\n")
        print(retry_answer)

        retry_validation = validate_answer(
            answer=retry_answer, query=question,
            query_type=strategy["query_type"], query_years=query_years,
        )

        answer = retry_answer
        results = retry_results
        validation = retry_validation

        rd = results["distances"][0]
        if rd:
            avg_similarity = round((1 - sum(rd) / len(rd)) * 100, 2)
            top_similarity = round((1 - min(rd)) * 100, 2)

    print("\n" + "=" * 60)

    return {
        "answer": answer, "results": results, "strategy": strategy,
        "retry_used": retry, "validation": validation,
        "avg_similarity": avg_similarity, "top_similarity": top_similarity,
        "retrieved_chunks": len(results["distances"][0]),
    }


if __name__ == "__main__":
    result = agentic_answer(
        question="Compare Tesla's revenue with Apple's in 2024",
        company=None,
    )
    print("\nFINAL ANSWER:")
    print(result["answer"])