import chromadb
import ollama

from sentence_transformers import SentenceTransformer

from query_classifier import classify_query
from router import build_retrieval_strategy

from query_parser import extract_years
from self_check import should_retry



# ============================================================
# LOAD MODELS + VECTOR DB
# ============================================================

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="data/vectorstore")


# ============================================================
# RETRIEVE USING DYNAMIC STRATEGY
# ============================================================

def retrieve_chunks_agentic(
    question,
    company=None,
    year=None
):

    # --------------------------------------------
    # Build retrieval strategy
    # --------------------------------------------

    strategy = build_retrieval_strategy(question)

    query_type = strategy["query_type"]
    top_k = strategy["top_k"]
    section = strategy["section"]

    print("\nRetrieval Strategy:")
    print(strategy)

    # --------------------------------------------
    # Build metadata filters
    # --------------------------------------------

    filters = []

    if company:
        filters.append({"company": company})

    query_years = extract_years(question)

    # Use query years if present
    if len(query_years) > 0:
        if len(query_years) == 1:
            filters.append({"year": query_years[0]})

        else:
            year_filters = []
            for y in query_years:
                year_filters.append({"year": y})

            filters.append({
                "$or": year_filters
            })

    # Otherwise use provided year
    elif year:
        filters.append({"year": year})
    
    if section:
        filters.append({"section": section})

    # ChromaDB filter logic
    if len(filters) == 0:
        where_filter = None

    elif len(filters) == 1:
        where_filter = filters[0]

    else:
        where_filter = {
            "$and": filters
        }

    # --------------------------------------------
    # Vector retrieval
    # --------------------------------------------

    collection = client.get_collection(name="section_chunks")

    query_embedding = embedding_model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_filter
    )

    return results, strategy


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
4. Be concise and factual.

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
# AGENTIC RAG PIPELINE
# ============================================================

def agentic_answer(
    question,
    company=None,
    year=None
):

    print("\n" + "="*60)
    print("AGENTIC RAG PIPELINE")
    print("="*60)

    print(f"\nQuestion: {question}")

    # --------------------------------------------
    # Retrieve dynamically
    # --------------------------------------------

    results, strategy = retrieve_chunks_agentic(
        question=question,
        company=company,
        year=year
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

    # --------------------------------------------
    # Generate answer
    # --------------------------------------------

    print("\nGenerating answer...\n")

    answer = generate_answer(question, results)

    print("\nInitial Answer:\n")
    print(answer)

    # ========================================================
    # SELF-CHECK
    # ========================================================

    retry = should_retry(results, answer)

    # ========================================================
    # RETRY IF NEEDED
    # ========================================================

    if retry:

        print("\nRetrying with broader retrieval...\n")

        # Retry with broader retrieval
        collection = client.get_collection(name="section_chunks")

        query_embedding = embedding_model.encode(question).tolist()

        retry_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=8
        )

        print("\nRetry Retrieved Sources:\n")

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

    print("\n" + "="*60)

    return answer

    
# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    agentic_answer(
        question="Compare Apple's risks between 2023 and 2024",
        company="AAPL",
        year=2024
    )