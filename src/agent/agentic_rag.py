"""
src/agent/agentic_rag.py

Agentic RAG pipeline with:
- Dynamic retrieval strategy routing
- Strict section filtering (preserved on retry)
- Multi-year metadata filtering
- Post-retrieval reranking
- Improved prompts
- Answer validation
"""

import chromadb
import os
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from src.agent.query_classifier import classify_query
from src.agent.router import build_retrieval_strategy
from src.agent.query_parser import extract_years
from src.agent.self_check import should_retry
from src.agent.answer_validator import validate_answer
from src.retrieval.reranker import rerank_results

# ============================================================
# LOAD API KEY + LLM
# ============================================================

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ============================================================
# LOAD MODELS + VECTOR DB
# ============================================================

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="data/vectorstore")


# ============================================================
# RETRIEVE USING DYNAMIC STRATEGY
# ============================================================

def retrieve_chunks_agentic(
    question: str,
    company: str = None,
    year: int = None
) -> tuple:
    """
    Dynamic retrieval using router strategy.

    Returns:
        (results dict, strategy dict)
    """

    strategy = build_retrieval_strategy(question)
    query_type = strategy["query_type"]
    top_k = strategy["top_k"]
    section = strategy["section"]

    print("\nRetrieval Strategy:")
    print(strategy)

    # --------------------------------------------------------
    # Build metadata filters
    # --------------------------------------------------------

    filters = []

    if company:
        filters.append({"company": company})

    query_years = extract_years(question)

    if len(query_years) > 0:
        if len(query_years) == 1:
            filters.append({"year": query_years[0]})
        else:
            # Multi-year OR filter
            filters.append({
                "$or": [{"year": y} for y in query_years]
            })
    elif year:
        filters.append({"year": year})

    # Strict section filtering based on query type
    if section:
        filters.append({"section": section})

    # ChromaDB filter construction
    if len(filters) == 0:
        where_filter = None
    elif len(filters) == 1:
        where_filter = filters[0]
    else:
        where_filter = {"$and": filters}

    # --------------------------------------------------------
    # Vector retrieval
    # --------------------------------------------------------

    collection = client.get_collection(name="section_chunks")
    query_embedding = embedding_model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_filter
    )

    return results, strategy


# ============================================================
# GENERATE ANSWER — GROQ
# ============================================================

def generate_answer(question: str, results: dict) -> str:
    """
    Generate answer using Groq (llama-3.3-70b-versatile) with retrieved context.
    Improved prompt for grounded synthesis.
    """

    context_parts = []

    for i in range(len(results["ids"][0])):
        doc = results["documents"][0][i][:1200]
        meta = results["metadatas"][0][i]
        context_parts.append(
            f"[Source {i+1}] "
            f"Company: {meta['company']} | "
            f"Year: {meta['year']} | "
            f"Section: {meta['section']}\n\n"
            f"{doc}"
        )

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are a senior financial analyst specializing in SEC 10-K filings.

Your task is to answer the question below using ONLY the provided source excerpts.

STRICT RULES:
1. Base your answer entirely on the provided sources. Do NOT use outside knowledge.
2. Cite every claim using the format (Source X) — e.g., (Source 1), (Source 2).
3. If multiple sources say the same thing, cite all relevant ones.
4. Synthesize information across sources; do not simply repeat them verbatim.
5. If the question is comparative (two years or two companies), explicitly address BOTH sides.
6. Include specific numbers, percentages, and dollar figures when present in the sources.
7. Only say evidence is insufficient if genuinely NO relevant information appears in ANY source.
   If partial information exists, present what is available and note what is missing.
8. Keep the answer concise and structured. Use bullet points for lists of risks or factors.

Context from SEC Filings:
{context}

Question: {question}

Answer:"""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.2,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"LLM generation failed: {str(e)}"


# ============================================================
# AGENTIC RAG PIPELINE
# ============================================================

def agentic_answer(
    question: str,
    company: str = None,
    year: int = None
) -> dict:
    """
    Full agentic pipeline:
      retrieve → rerank → generate → self-check → retry (section-preserving) → validate

    Returns:
        dict with keys: answer, results, strategy, retry_used,
                        avg_similarity, top_similarity, retrieved_chunks,
                        validation.
    """

    print("\n" + "="*60)
    print("AGENTIC RAG PIPELINE")
    print("="*60)
    print(f"\nQuestion: {question}")

    # --------------------------------------------------------
    # Retrieve dynamically
    # --------------------------------------------------------

    results, strategy = retrieve_chunks_agentic(
        question=question,
        company=company,
        year=year
    )

    # --------------------------------------------------------
    # Rerank
    # --------------------------------------------------------

    results = rerank_results(question, results)

    distances = results["distances"][0]
    avg_distance = round(sum(distances) / len(distances), 4)
    avg_similarity = round((1 - avg_distance) * 100, 2)
    top_similarity = round((1 - min(distances)) * 100, 2)

    print("\nRetrieved Sources (after reranking):\n")
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        dist = results["distances"][0][i]
        print(
            f"[{i+1}] "
            f"{meta['company']} | "
            f"Year: {meta['year']} | "
            f"{meta['section']} | "
            f"Distance: {dist:.4f}"
        )

    # --------------------------------------------------------
    # Generate answer
    # --------------------------------------------------------

    print("\nGenerating answer...\n")
    answer = generate_answer(question, results)
    print("\nInitial Answer:\n")
    print(answer)

    # --------------------------------------------------------
    # Self-check
    # --------------------------------------------------------

    retry = should_retry(results, answer)
    retry_used = False

    # --------------------------------------------------------
    # Retry — PRESERVES original section filter
    # --------------------------------------------------------

    if retry:

        print("\nRetrying with broader top_k (section preserved)...\n")

        section = strategy.get("section")
        query_years = extract_years(question)

        retry_filters = []

        if company:
            retry_filters.append({"company": company})

        if len(query_years) > 0:
            if len(query_years) == 1:
                retry_filters.append({"year": query_years[0]})
            else:
                retry_filters.append({
                    "$or": [{"year": y} for y in query_years]
                })
        elif year:
            retry_filters.append({"year": year})

        # IMPORTANT: preserve section filter, do NOT drop it
        if section:
            retry_filters.append({"section": section})

        if len(retry_filters) == 0:
            retry_where = None
        elif len(retry_filters) == 1:
            retry_where = retry_filters[0]
        else:
            retry_where = {"$and": retry_filters}

        collection = client.get_collection(name="section_chunks")
        query_embedding = embedding_model.encode(question).tolist()

        retry_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=8,          # Broader k
            where=retry_where
        )

        # Rerank retry results
        retry_results = rerank_results(question, retry_results)

        print("\nRetry Retrieved Sources (after reranking):\n")
        for i in range(len(retry_results["ids"][0])):
            meta = retry_results["metadatas"][0][i]
            dist = retry_results["distances"][0][i]
            print(
                f"[{i+1}] "
                f"{meta['company']} | "
                f"Year: {meta['year']} | "
                f"{meta['section']} | "
                f"Distance: {dist:.4f}"
            )

        print("\nGenerating Retry Answer...\n")
        retry_answer = generate_answer(question, retry_results)
        print("\nRetry Answer:\n")
        print(retry_answer)

        answer = retry_answer
        results = retry_results
        retry_used = True

        # Recompute similarity stats from retry results
        distances = results["distances"][0]
        avg_distance = round(sum(distances) / len(distances), 4)
        avg_similarity = round((1 - avg_distance) * 100, 2)
        top_similarity = round((1 - min(distances)) * 100, 2)

    # --------------------------------------------------------
    # Answer validation
    # --------------------------------------------------------

    validation = validate_answer(question, answer, results)

    print("\nAnswer Validation:")
    for k, v in validation.items():
        print(f"  {k}: {v}")

    print("\n" + "="*60)

    return {
        "answer": answer,
        "results": results,
        "strategy": strategy,
        "retry_used": retry_used,
        "avg_similarity": avg_similarity,
        "top_similarity": top_similarity,
        "retrieved_chunks": len(distances),
        "validation": validation,
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    agentic_answer(
        question="Compare Apple's risks between 2023 and 2024",
        company="AAPL"
    )