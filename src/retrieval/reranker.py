"""
src/retrieval/reranker.py   (cross-encoder + optional table boost)

Cross-encoder reranker.

WHY THE CROSS-ENCODER (unchanged rationale)
-------------------------------------------
A cross-encoder reads the (query, chunk) PAIR jointly and scores relevance
directly. Use this as the FINAL stage: fuse (dense+BM25) -> rerank -> top_k.
Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (~80MB, fine for HF Spaces).

WHAT'S NEW: `table_boost`
-------------------------
ms-marco was trained on natural-language passages, so for numeric queries
it tends to rank narrative prose ABOVE the financial-statement table that
actually holds the number. `table_boost` adds a small constant to the
cross-encoder score of any chunk whose metadata has content_type=="table".

It is OPT-IN: callers pass table_boost>0 ONLY for numeric/comparative
queries. The default (0.0) reproduces the previous behaviour exactly, so
descriptive / risk / summary ranking is unchanged.
"""

from sentence_transformers import CrossEncoder

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_model = None


def _get_model():
    global _model
    if _model is None:
        _model = CrossEncoder(_MODEL_NAME)
    return _model


def rerank_results(question: str, results: dict, top_k: int = None,
                   table_boost: float = 0.0) -> dict:
    """
    Rerank a Chroma-style result dict with a cross-encoder.

    Args:
        question:    user query.
        results:     Chroma-style dict (ids/documents/metadatas/distances).
        top_k:       if set, truncate to this many after reranking.
        table_boost: additive bonus (in ms-marco logit units) applied to the
                     score of any chunk whose metadata content_type=="table".
                     0.0 = no change (default). ~2.0 reliably surfaces the
                     financial-statement table for numeric questions.

    Returns:
        New Chroma-style dict ordered by (boosted) cross-encoder score (desc).
        `distances` = 1 - min-max-normalized score (smaller = better).
    """
    if not results or not results.get("ids") or not results["ids"][0]:
        return results

    documents = results["documents"][0]
    ids = results["ids"][0]
    metadatas = results["metadatas"][0]

    model = _get_model()
    # Truncate very long table chunks for the cross-encoder input window.
    pairs = [[question, doc[:2000]] for doc in documents]
    raw_scores = model.predict(pairs).tolist()

    # Apply the table boost (no-op when table_boost == 0.0).
    scores = []
    for i, s in enumerate(raw_scores):
        is_table = metadatas[i].get("content_type") == "table"
        scores.append(s + (table_boost if is_table else 0.0))

    order = sorted(range(len(scores)), key=lambda i: scores[i],
                   reverse=True)
    if top_k:
        order = order[:top_k]

    # Min-max normalize (boosted) scores to pseudo-distances in [0,1].
    s_min = min(scores)
    s_max = max(scores)
    rng = (s_max - s_min) or 1.0

    out = {
        "ids": [[ids[i] for i in order]],
        "documents": [[documents[i] for i in order]],
        "metadatas": [[metadatas[i] for i in order]],
        "distances": [[1.0 - ((scores[i] - s_min) / rng) for i in order]],
    }

    print("\nReranking (cross-encoder"
          f"{', table_boost=%.1f' % table_boost if table_boost else ''}):")
    for rank, i in enumerate(order):
        m = metadatas[i]
        tag = " [TABLE]" if m.get("content_type") == "table" else ""
        print(f"  Rank {rank+1}: {m['company']} | {m['year']} | "
              f"{m['section']}{tag} | ce_score={scores[i]:.3f}")

    return out


if __name__ == "__main__":
    fake = {
        "ids": [["id1", "id2", "id3"]],
        "documents": [[
            "Apple faces significant competition in all its markets.",
            "[TABLE] Total net sales 391,035 | Net income 93,736 [/TABLE]",
            "Risk factors include supply chain and geopolitical instability.",
        ]],
        "metadatas": [[
            {"company": "AAPL", "year": 2024, "section": "Business",
             "content_type": "text"},
            {"company": "AAPL", "year": 2024, "section": "Financial Statements",
             "content_type": "table"},
            {"company": "AAPL", "year": 2024, "section": "Risk Factors",
             "content_type": "text"},
        ]],
        "distances": [[0.42, 0.38, 0.31]],
    }
    print("WITHOUT boost:")
    out = rerank_results("What were Apple's total net sales in 2024?", fake)
    print("\nWITH table_boost=2.0:")
    out = rerank_results("What were Apple's total net sales in 2024?", fake,
                         table_boost=2.0)