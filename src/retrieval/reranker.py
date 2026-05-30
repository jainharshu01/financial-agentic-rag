"""
src/retrieval/reranker.py   (REPLACES the old version)

Cross-encoder reranker.

WHY THIS CHANGED
----------------
The old reranker re-scored candidates with the SAME bi-encoder
(all-MiniLM-L6-v2) that produced the original ranking — it mostly
re-derived the same cosine order, adding latency for ~no signal, and it
wasn't even imported into agentic_rag.py.

A cross-encoder reads the (query, chunk) PAIR jointly and scores
relevance directly. This is what actually moves precision (benchmarks
report up to ~28% nDCG@10 gains). cross-encoder/ms-marco-MiniLM-L-6-v2
is ~80MB — fine for HuggingFace Spaces.

Use this as the FINAL stage: fuse (dense+BM25) -> rerank -> top_k.
"""

from sentence_transformers import CrossEncoder

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_model = None


def _get_model():
    global _model
    if _model is None:
        _model = CrossEncoder(_MODEL_NAME)
    return _model


def rerank_results(question: str, results: dict, top_k: int = None) -> dict:
    """
    Rerank a Chroma-style result dict with a cross-encoder.

    Args:
        question: user query.
        results:  Chroma-style dict (ids/documents/metadatas/distances).
        top_k:    if set, truncate to this many after reranking.

    Returns:
        New Chroma-style dict ordered by cross-encoder score (desc).
        `distances` = 1 - sigmoid-ish normalized score (smaller better).
    """
    if not results or not results.get("ids") or not results["ids"][0]:
        return results

    documents = results["documents"][0]
    ids = results["ids"][0]
    metadatas = results["metadatas"][0]

    model = _get_model()
    # Truncate very long table chunks for the cross-encoder input window.
    pairs = [[question, doc[:2000]] for doc in documents]
    scores = model.predict(pairs).tolist()

    order = sorted(range(len(scores)), key=lambda i: scores[i],
                   reverse=True)
    if top_k:
        order = order[:top_k]

    # Min-max normalize scores to pseudo-distances in [0,1].
    s_min = min(scores)
    s_max = max(scores)
    rng = (s_max - s_min) or 1.0

    out = {
        "ids": [[ids[i] for i in order]],
        "documents": [[documents[i] for i in order]],
        "metadatas": [[metadatas[i] for i in order]],
        "distances": [[1.0 - ((scores[i] - s_min) / rng) for i in order]],
    }

    print("\nReranking (cross-encoder):")
    for rank, i in enumerate(order):
        m = metadatas[i]
        print(f"  Rank {rank+1}: {m['company']} | {m['year']} | "
              f"{m['section']} | ce_score={scores[i]:.3f}")

    return out


if __name__ == "__main__":
    fake = {
        "ids": [["id1", "id2", "id3"]],
        "documents": [[
            "Apple faces significant competition in all its markets.",
            "Tesla FY2024 revenue was approximately 97.7 billion dollars.",
            "Risk factors include supply chain and geopolitical instability.",
        ]],
        "metadatas": [[
            {"company": "AAPL", "year": 2024, "section": "Business"},
            {"company": "TSLA", "year": 2024, "section": "Financial Statements"},
            {"company": "AAPL", "year": 2024, "section": "Risk Factors"},
        ]],
        "distances": [[0.42, 0.38, 0.31]],
    }
    out = rerank_results("What are Apple's major risk factors?", fake)
    print("\nReranked order:")
    for d in out["documents"][0]:
        print(" -", d[:70])