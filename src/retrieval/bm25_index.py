"""
src/retrieval/bm25_index.py   (NEW)

Lexical (sparse) retrieval over the same chunks as the vector store.

WHY
---
A 2026 text-and-table financial RAG benchmark found BM25 OUTPERFORMS
dense retrieval on financial documents, because exact tokens — "diluted
EPS", "FY2024", "Item 1A", "$391,035" — are what these queries hinge on,
and embeddings blur exactly those. We add BM25 as a second channel and
fuse it with dense results (see fusion.py).

This index is built in-memory at startup from the chunk JSON (the same
file build_vectorstore.py uses), so it stays perfectly in sync with the
vector store and needs no separate persistence. ~1,200 chunks is trivial.

Financial synonym expansion is applied HERE (query side only), per the
research finding that expansion helps lexical, not dense, retrieval.
"""

import json
import re

from rank_bm25 import BM25Okapi

from src.agent.financial_synonyms import expand_query_terms


def _tokenize(text: str) -> list:
    return re.findall(r"[a-z0-9$%\.]+", text.lower())


class BM25Index:
    def __init__(self, chunk_file="data/chunks/section_chunks.json"):
        with open(chunk_file, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        self.ids = [c["chunk_id"] for c in self.chunks]
        self.documents = [c["text"] for c in self.chunks]
        self.metadatas = [
            {
                "company": c["company"],
                "year": c["year"],
                "filing_id": c["filing_id"],
                "chunk_method": c["chunk_method"],
                "section": c["section"],
                "content_type": c.get("content_type", "text"),
            }
            for c in self.chunks
        ]

        tokenized = [_tokenize(d) for d in self.documents]
        self.bm25 = BM25Okapi(tokenized)

    def query(self, question: str, top_k: int = 20,
              where: dict = None) -> dict:
        """
        Return a Chroma-style result dict so it drops into the existing
        pipeline. Applies optional metadata filtering and financial
        synonym expansion on the query.
        """
        terms = _tokenize(question) + expand_query_terms(question)
        scores = self.bm25.get_scores(terms)

        # Rank all, then apply metadata filter, then take top_k.
        order = sorted(range(len(scores)), key=lambda i: scores[i],
                       reverse=True)

        ids, docs, metas, dists = [], [], [], []
        for i in order:
            meta = self.metadatas[i]
            if where and not _matches(meta, where):
                continue
            ids.append(self.ids[i])
            docs.append(self.documents[i])
            metas.append(meta)
            # Convert score to a pseudo-distance for uniformity.
            dists.append(1.0 / (1.0 + scores[i]))
            if len(ids) >= top_k:
                break

        return {
            "ids": [ids],
            "documents": [docs],
            "metadatas": [metas],
            "distances": [dists],
        }


def _matches(meta: dict, where: dict) -> bool:
    """
    Minimal re-implementation of the subset of Chroma `where` operators
    this project uses: equality, {"$and":[...]}, {"$or":[...]}.
    """
    if "$and" in where:
        return all(_matches(meta, sub) for sub in where["$and"])
    if "$or" in where:
        return any(_matches(meta, sub) for sub in where["$or"])
    for key, val in where.items():
        if meta.get(key) != val:
            return False
    return True


# Singleton so we build the index once per process.
_INDEX = None


def get_bm25_index(chunk_file="data/chunks/section_chunks.json") -> BM25Index:
    global _INDEX
    if _INDEX is None:
        _INDEX = BM25Index(chunk_file)
    return _INDEX


if __name__ == "__main__":
    idx = get_bm25_index()
    res = idx.query("Apple net sales 2024",
                    top_k=5, where={"company": "AAPL"})
    for i in range(len(res["ids"][0])):
        m = res["metadatas"][0][i]
        print(f"{m['company']} | {m['year']} | {m['section']} | "
              f"{m['content_type']} | dist={res['distances'][0][i]:.4f}")