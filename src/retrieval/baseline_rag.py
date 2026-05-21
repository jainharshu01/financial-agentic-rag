import chromadb
import ollama
from sentence_transformers import SentenceTransformer

# ============================================================
# LOAD EMBEDDING MODEL + VECTOR DB
# ============================================================

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="data/vectorstore")


# ============================================================
# RETRIEVE CHUNKS
# ============================================================

def retrieve_chunks(question, company=None, year=None, top_k=3):
    """
    Static semantic retrieval.
    No routing.
    No adaptive logic.
    """

    collection = client.get_collection(name="section_chunks")

    query_embedding = embedding_model.encode(question).tolist()

    # Build metadata filter

    filters = []

    if company:
        filters.append({"company": company})

    if year:
        filters.append({"year": year})

    if len(filters) == 0:
        where_filter = None

    elif len(filters) == 1:
        where_filter = filters[0]

    else:
        where_filter = {
            "$and": filters
        }

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_filter
    )

    return results


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(question, results):

    context_parts = []

    for i in range(len(results["ids"][0])):

        doc = results["documents"][0][i][:600]
        meta = results["metadatas"][0][i]

        context_parts.append(
            f"[Source {i+1}] "
            f"Company: {meta['company']} | "
            f"Year: {meta['year']} | "
            f"Section: {meta['section']} \n\n"
            f"{doc}"
        )

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""
You are a financial document analyst.

Answer the user's question using ONLY the provided SEC filing context.

RULES:
1. Do NOT use outside knowledge.
2. Cite sources using [Source X].
3. If evidence is insufficient, say:
   "Insufficient evidence in the provided documents."
4. Be concise but factual.

Context:
{context}

Question:
{question}

Answer:
"""

    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


# ============================================================
# BASELINE RAG PIPELINE
# ============================================================

def baseline_answer(question, company=None, year=None, top_k=3):

    print("\n" + "="*60)
    print("BASELINE STATIC RAG")
    print("="*60)

    print(f"\nQuestion: {question}")
    print(f"Company: {company}")
    print(f"Year: {year}")

    # Retrieve
    results = retrieve_chunks(
        question=question,
        company=company,
        year=year,
        top_k=top_k
    )

    print("\nRetrieved Sources:\n")

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
        "retry_used": False
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    baseline_answer(
        question="What were Apple's major risk factors?",
        company="AAPL",
        year=2024
    )