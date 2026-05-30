"""
src/retrieval/build_vectorstore.py   (REPLACES the old version)

CHANGES
-------
- Embedding model switched to BAAI/bge-small-en-v1.5 (384-dim, 512-token
  context) so our ~400-token chunks are fully encoded instead of being
  silently truncated at 256 tokens by all-MiniLM-L6-v2.
- Stores the new `content_type` metadata field ("text" | "table").
- bge models benefit from a retrieval instruction prefix on the QUERY
  side only; documents are embedded as-is. We add that prefix at query
  time (see bm25_index / retrieval), NOT here.
"""

import os
import json

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"


def build_vectorstore(chunk_file, collection_name):
    with open(chunk_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks from {chunk_file}")

    print(f"Loading embedding model: {EMBED_MODEL_NAME} ...")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    db_path = "data/vectorstore"
    os.makedirs(db_path, exist_ok=True)
    client = chromadb.PersistentClient(path=db_path)

    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    batch_size = 100
    for i in tqdm(range(0, len(chunks), batch_size),
                  desc="Embedding & storing"):
        batch = chunks[i:i + batch_size]
        texts = [c["text"] for c in batch]
        ids = [c["chunk_id"] for c in batch]
        metadatas = [
            {
                "company": c["company"],
                "year": c["year"],
                "filing_id": c["filing_id"],
                "chunk_method": c["chunk_method"],
                "section": c["section"],
                # NEW: lets retrieval boost/identify table chunks.
                "content_type": c.get("content_type", "text"),
            }
            for c in batch
        ]
        embeddings = model.encode(
            texts, show_progress_bar=False, normalize_embeddings=True
        ).tolist()
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    print(f"Done! '{collection_name}' has {collection.count()} documents.")
    return collection


if __name__ == "__main__":
    print("=" * 50)
    print("Building vectorstore for FIXED chunks")
    print("=" * 50)
    build_vectorstore("data/chunks/fixed_chunks.json", "fixed_chunks")

    print()
    print("=" * 50)
    print("Building vectorstore for SECTION chunks")
    print("=" * 50)
    build_vectorstore("data/chunks/section_chunks.json", "section_chunks")

    print("\nAll vectorstores built!")