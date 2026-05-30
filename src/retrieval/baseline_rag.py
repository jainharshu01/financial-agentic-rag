"""
src/retrieval/baseline_rag.py   (REPLACES the old version)

Baseline static RAG. Kept deliberately simple as the comparison point.

ONLY CHANGES vs old version:
- Embedding model -> BAAI/bge-small-en-v1.5 (matches the new vectorstore).
- bge wants a short instruction prefix on the QUERY (not documents):
  "Represent this sentence for searching relevant passages:".
- normalize_embeddings=True (cosine space).
Everything else (filters, auto-detect, generation) is unchanged so the
baseline-vs-agentic comparison stays fair.
"""

import os
import chromadb

from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

from src.agent.company_parser import extract_companies

load_dotenv()

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

embedding_model = SentenceTransformer(EMBED_MODEL_NAME)
client = chromadb.PersistentClient(path="data/vectorstore")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def _embed_query(question):
    return embedding_model.encode(
        QUERY_PREFIX + question, normalize_embeddings=True
    ).tolist()


def retrieve_chunks(question, company=None, year=None, section=None, top_k=5):
    collection = client.get_collection(name="section_chunks")
    query_embedding = _embed_query(question)

    if company:
        target_companies = [company]
    else:
        target_companies = extract_companies(question)

    filters = []
    if target_companies:
        if len(target_companies) == 1:
            filters.append({"company": target_companies[0]})
        else:
            filters.append({"$or": [{"company": c} for c in target_companies]})
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

    return collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_filter,
    )


def generate_answer(question, results):
    context_parts = []
    for i in range(len(results["ids"][0])):
        doc = results["documents"][0][i][:600]
        meta = results["metadatas"][0][i]
        context_parts.append(
            f"[Source {i+1}] Company: {meta['company']} | "
            f"Year: {meta['year']} | Section: {meta['section']} \n\n{doc}"
        )
    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are a financial document analyst.

Answer the user's question using ONLY the provided SEC filing context.

RULES:
1. Do NOT use outside knowledge.
2. Cite sources using (Source X) format — always include the parentheses.
3. For cross-company comparisons, mention EACH company and cite sources from each.
4. Include specific numbers when relevant.
5. If evidence is insufficient, say: "Insufficient evidence in the provided documents."

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


def baseline_answer(question, company=None, year=None, section=None, top_k=5):
    print("\n" + "=" * 60)
    print("BASELINE STATIC RAG")
    print("=" * 60)
    print(f"\nQuestion: {question}")
    print(f"Company: {company or '[auto-detect]'}")
    print(f"Year: {year}")
    print(f"Section: {section}")

    results = retrieve_chunks(
        question=question, company=company,
        year=year, section=section, top_k=top_k,
    )

    distances = results["distances"][0]
    if not distances:
        return {
            "answer": "No relevant documents found.",
            "results": results, "retry_used": False,
            "avg_similarity": 0, "top_similarity": 0, "retrieved_chunks": 0,
        }

    avg_distance = round(sum(distances) / len(distances), 4)
    avg_similarity = round((1 - avg_distance) * 100, 2)
    top_similarity = round((1 - min(distances)) * 100, 2)

    print("\nRetrieved Sources:\n")
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        dist = results["distances"][0][i]
        print(f"[{i+1}] {meta['company']} | Year: {meta['year']} | "
              f"{meta['section']} | Distance: {dist:.4f}")

    print("\nGenerating answer...\n")
    answer = generate_answer(question, results)
    print(answer)
    print("\n" + "=" * 60)

    return {
        "answer": answer, "results": results, "retry_used": False,
        "avg_similarity": avg_similarity, "top_similarity": top_similarity,
        "retrieved_chunks": len(distances),
    }


if __name__ == "__main__":
    baseline_answer(
        question="Compare Tesla's revenue with Apple's in 2024",
        company=None,
    )