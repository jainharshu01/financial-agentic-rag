"""
src/retrieval/fusion.py   (NEW)

Reciprocal Rank Fusion (RRF) for combining dense + BM25 result lists.

WHY RRF (not weighted score sum)
--------------------------------
BM25 scores and cosine distances live on incompatible scales. RRF
operates on RANKS, not raw scores, so it sidesteps normalization entirely
and is robust across score scales (Weaviate, Elasticsearch, and the
finance benchmark all recommend it as the default). Standard smoothing
constant k = 60.

A document's RRF score = sum over each list of 1 / (k + rank_in_list).
Documents appearing high in either list rise to the top.
"""


def reciprocal_rank_fusion(dense_results: dict,
                           sparse_results: dict,
                           top_k: int = 6,
                           k: int = 60) -> dict:
    """
    Fuse two Chroma-style result dicts into one, ranked by RRF.

    Args:
        dense_results:  result dict from the vector store.
        sparse_results: result dict from BM25Index.query.
        top_k:          number of fused results to return.
        k:              RRF smoothing constant (60 is standard).

    Returns:
        A Chroma-style result dict (single outer list), RRF-ordered.
        `distances` is set to (1 - rrf_score_normalized) so that smaller
        is still "better", matching the rest of the pipeline.
    """
    scores = {}        # chunk_id -> rrf score
    meta_by_id = {}    # chunk_id -> (doc, meta)

    def ingest(results):
        if not results or not results.get("ids") or not results["ids"][0]:
            return
        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        for rank, cid in enumerate(ids):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            if cid not in meta_by_id:
                meta_by_id[cid] = (docs[rank], metas[rank])

    ingest(dense_results)
    ingest(sparse_results)

    if not scores:
        return {"ids": [[]], "documents": [[]],
                "metadatas": [[]], "distances": [[]]}

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    ranked = ranked[:top_k]

    max_score = ranked[0][1] if ranked else 1.0

    ids, docs, metas, dists = [], [], [], []
    for cid, sc in ranked:
        doc, meta = meta_by_id[cid]
        ids.append(cid)
        docs.append(doc)
        metas.append(meta)
        # Normalize to a pseudo-distance in [0,1], smaller = better.
        dists.append(1.0 - (sc / max_score))

    return {
        "ids": [ids],
        "documents": [docs],
        "metadatas": [metas],
        "distances": [dists],
    }


if __name__ == "__main__":
    dense = {
        "ids": [["a", "b", "c"]],
        "documents": [["da", "db", "dc"]],
        "metadatas": [[{"company": "AAPL"}] * 3],
        "distances": [[0.1, 0.2, 0.3]],
    }
    sparse = {
        "ids": [["c", "a", "d"]],
        "documents": [["dc", "da", "dd"]],
        "metadatas": [[{"company": "AAPL"}] * 3],
        "distances": [[0.05, 0.15, 0.25]],
    }
    fused = reciprocal_rank_fusion(dense, sparse, top_k=4)
    print("Fused order:", fused["ids"][0])