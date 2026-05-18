import os
import ollama
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import google.generativeai as genai

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Load models
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load vector store
client = chromadb.PersistentClient(path="data/vectorstore")


def retrieve_chunks(question, collection_name="section_chunks", company=None, top_k=5):
    """Retrieve top-k relevant chunks for a question."""
    collection = client.get_collection(name=collection_name)
    query_embedding = embedding_model.encode(question).tolist()
    
    # Build filter if company specified
    # Build filter if company specified
    where_filter = None

    if company:
        where_filter = {"company": company}
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_filter
    )
    
    return results


def generate_answer(question, results):
    """Generate answer using Gemini with retrieved context."""
    
    # Build context from retrieved chunks
    context_parts = []
    for i in range(len(results["ids"][0])):
        doc = results["documents"][0][i][:1200]
        meta = results["metadatas"][0][i]
        context_parts.append(
            f"[Source {i+1}] Company: {meta['company']} | "
            f"Section: {meta['section']} | "
            f"Filing: {meta['filing_id']}\n{doc}"
        )
    
    context = "\n\n---\n\n".join(context_parts)
    
    # Build prompt
    prompt = f"""You are a financial document analyst. Answer the question using ONLY the provided context from SEC filings.

Rules:
1. Answer ONLY from the given context. Do not use outside knowledge.
2. Cite your sources using [Source X] notation.
3. If the context does not contain enough information, say "Insufficient evidence in the provided documents."
4. Be specific and include numbers/facts when available.

Context:
{context}

Question: {question}

Answer:"""
    
    # Generate answer
    try:
        response = ollama.chat(
            model="llama3.2",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response["message"]["content"]
    except Exception as e:
        return f"LLM generation failed: {str(e)}"

def rag_answer(question, company=None, collection_name="section_chunks", top_k=2):
    """Full RAG pipeline: retrieve + generate."""
    
    print(f"Question: {question}")
    print(f"Company filter: {company}")
    print(f"Collection: {collection_name}")
    print("-" * 50)
    
    # Retrieve
    results = retrieve_chunks(question, collection_name, company, top_k)
    
    # Show retrieved sources
    print("\nRetrieved Sources:")
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        dist = results["distances"][0][i]
        print(
            f"  [{i+1}] "
            f"{meta['company']} | "
            f"Year: {meta['year']} | "
            f"{meta['section']} | "
            f"Distance: {dist:.4f}"
        )
    # Generate answer
    print("\nGenerating answer...")
    answer = generate_answer(question, results)
    
    print(f"\nAnswer:\n{answer}")
    print("=" * 50)
    
    return answer


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("TEST 1: Apple Risk Factors")
    print("=" * 50)
    rag_answer("What are the major risk factors for Apple?", company="AAPL")