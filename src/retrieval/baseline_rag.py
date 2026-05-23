"""
src/retrieval/baseline_rag.py

Static RAG pipeline.

Changes from original:
- Improved prompt that encourages grounded synthesis instead of "Insufficient evidence".
- Post-retrieval reranking via reranker.py before answer generation.
- Citation format standardized to (Source X).
- Strict section filtering in retrieve_chunks().
"""

import chromadb
import os
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from src.retrieval.reranker import rerank_results

# ============================================================
# LOAD API KEY + LLM
# ============================================================

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ============================================================
# LOAD EMBEDDING MODEL + VECTOR DB
# ============================================================

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="data/vectorstore")


# ============================================================
# RETRIEVE CHUNKS
# ============================================================

def retrieve_chunks(
    question: str,
    company: str = None,
    year: int = None,
    section: str = None,
    top_k: int = 3
) -> dict:
    """
    Semantic retrieval with optional metadata filtering.

    Args:
        question: User question string.
        company:  Ticker symbol e.g. 'AAPL'. Optional.
        year:     Filing year as int. Optional.
        section:  SEC section name e.g. 'Risk Factors'. Optional.
        top_k:    Number of results to return.

    Returns:
        ChromaDB query results dict.
    """

    collection = client.get_collection(name="section_chunks")
    query_embedding = embedding_model.encode(question).tolist()

    filters = []

    if company:
        filters.append({"company": company})

    if year:
        filters.append({"year": year})

    if section:
        filters.append({"section": section})

    if len(filters) == 0:
        where_filter = None
    elif len(filters) == 1:
        where_filter = filters[0]
    else:
        where_filter = {"$and": filters}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_filter
    )

    return results


# ============================================================
# GENERATE ANSWER — GROQ
# ============================================================

def generate_answer(question: str, results: dict) -> str:
    """
    Generate answer using Groq (llama-3.3-70b-versatile) with retrieved context.

    Prompt improvements over the original:
    - Instructs model to synthesize, not merely report absence.
    - Uses (Source X) citation format.
    - Encourages numeric precision and comparative structure.
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
# BASELINE RAG PIPELINE
# ============================================================

def baseline_answer(
    question: str,
    company: str = None,
    year: int = None,
    section: str = None,
    top_k: int = 3
) -> dict:
    """
    Full baseline RAG pipeline: retrieve → rerank → generate.
    If section is not passed, it is derived from the query type automatically.
    """

    # Auto-derive section from query type if not explicitly provided
    if section is None:
        from src.agent.router import build_retrieval_strategy
        strategy = build_retrieval_strategy(question)
        section = strategy.get("section")

    print("\n" + "="*60)
    print("BASELINE STATIC RAG")
    print("="*60)
    print(f"\nQuestion: {question}")
    print(f"Company: {company} | Year: {year} | Section: {section}")

    # Retrieve
    results = retrieve_chunks(
        question=question,
        company=company,
        year=year,
        section=section,
        top_k=top_k
    )

    # Rerank
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

    # Generate
    print("\nGenerating answer...\n")
    answer = generate_answer(question, results)
    print(answer)
    print("\n" + "="*60)

    return {
        "answer": answer,
        "results": results,
        "retry_used": False,
        "avg_similarity": avg_similarity,
        "top_similarity": top_similarity,
        "retrieved_chunks": len(distances),
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    baseline_answer(
        question="What were Apple's major risk factors?",
        company="AAPL",
        year=2024
        # section is intentionally omitted — auto-derived from router
    )