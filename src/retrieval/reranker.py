"""
src/retrieval/reranker.py

Post-retrieval reranker using sentence-transformers cosine similarity
between the query and each retrieved chunk.

Design notes:
- We intentionally avoid a cross-encoder (e.g. cross-encoder/ms-marco-*)
  so we do NOT add a second heavy model to the stack.
- Instead we re-score using the same bi-encoder already loaded project-wide
  (all-MiniLM-L6-v2).  This is a lightweight but effective approach:
  the bi-encoder embedding of the query vs. each chunk gives a meaningful
  re-rank signal when the original vector search has already narrowed
  the candidate set to ~3-8 results.
- Returns the same ChromaDB result dict structure so callers need no changes
  beyond calling rerank_results() before generate_answer().
"""

from sentence_transformers import SentenceTransformer, util
import numpy as np

# ============================================================
# LOAD EMBEDDING MODEL (singleton pattern — import once)
# ============================================================

_rerank_model = None


def _get_model():
    global _rerank_model
    if _rerank_model is None:
        _rerank_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _rerank_model


# ============================================================
# RERANK
# ============================================================

def rerank_results(question: str, results: dict) -> dict:
    """
    Rerank ChromaDB query results by cosine similarity
    between the query embedding and each chunk embedding.

    Args:
        question: The user's original question string.
        results:  ChromaDB query() result dict with keys:
                  ids, documents, metadatas, distances
                  (all wrapped in a single-element outer list).

    Returns:
        A new results dict with the same structure,
        sorted descending by rerank similarity score.
    """

    if not results or not results.get("ids") or not results["ids"][0]:
        return results

    model = _get_model()

    documents = results["documents"][0]
    ids = results["ids"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    n = len(documents)

    # --------------------------------------------------------
    # Encode query once + encode all chunk texts
    # --------------------------------------------------------

    query_embedding = model.encode(question, convert_to_tensor=True)

    # Truncate chunks to 512 tokens to stay within model limits
    truncated_docs = [doc[:2000] for doc in documents]
    chunk_embeddings = model.encode(truncated_docs, convert_to_tensor=True)

    # --------------------------------------------------------
    # Compute cosine similarities
    # --------------------------------------------------------

    similarities = util.cos_sim(query_embedding, chunk_embeddings)[0]  # shape: (n,)
    similarity_scores = similarities.cpu().numpy().tolist()

    # --------------------------------------------------------
    # Sort indices by descending similarity
    # --------------------------------------------------------

    sorted_indices = np.argsort(similarity_scores)[::-1].tolist()

    # --------------------------------------------------------
    # Rebuild result dict in sorted order
    # --------------------------------------------------------

    reranked = {
        "ids": [[ids[i] for i in sorted_indices]],
        "documents": [[documents[i] for i in sorted_indices]],
        "metadatas": [[metadatas[i] for i in sorted_indices]],
        # Replace original distances with rerank-derived distances (1 - sim)
        "distances": [[1.0 - similarity_scores[i] for i in sorted_indices]],
    }

    print("\nReranking Results:")
    for rank, i in enumerate(sorted_indices):
        meta = metadatas[i]
        print(
            f"  Rank {rank+1}: "
            f"{meta['company']} | "
            f"Year: {meta['year']} | "
            f"{meta['section']} | "
            f"Rerank Sim: {similarity_scores[i]:.4f}"
        )

    return reranked


# ============================================================
# TESTS
# ============================================================

if __name__ == "__main__":

    # Minimal smoke test — does not require ChromaDB
    fake_results = {
        "ids": [["id1", "id2", "id3"]],
        "documents": [
            [
                "Apple faces significant competition in all its markets.",
                "Tesla's revenue for FY2024 was approximately 97.7 billion dollars.",
                "Risk factors include supply chain disruptions and geopolitical instability.",
            ]
        ],
        "metadatas": [
            [
                {"company": "AAPL", "year": 2024, "section": "Business"},
                {"company": "TSLA", "year": 2024, "section": "Financial Statements"},
                {"company": "AAPL", "year": 2024, "section": "Risk Factors"},
            ]
        ],
        "distances": [[0.42, 0.38, 0.31]],
    }

    question = "What are Apple's major risk factors?"
    reranked = rerank_results(question, fake_results)

    print("\nReranked order:")
    for i, doc in enumerate(reranked["documents"][0]):
        print(f"  {i+1}. {doc[:80]}...")