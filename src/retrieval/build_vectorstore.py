import json
import os
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def build_vectorstore(chunk_file, collection_name):
    """Load chunks, generate embeddings, store in ChromaDB."""
    
    # Load chunks
    with open(chunk_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    print(f"Loaded {len(chunks)} chunks from {chunk_file}")
    
    # Load embedding model
    print("Loading embedding model (this may take a minute first time)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Create ChromaDB client (persistent storage)
    db_path = "data/vectorstore"
    os.makedirs(db_path, exist_ok=True)
    client = chromadb.PersistentClient(path=db_path)
    
    # Delete collection if it already exists (fresh start)
    try:
        client.delete_collection(name=collection_name)
    except:
        pass
    
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    
    # Process in batches (ChromaDB has limits)
    batch_size = 100
    
    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding & storing"):
        batch = chunks[i:i + batch_size]
        
        texts = [c["text"] for c in batch]
        ids = [c["chunk_id"] for c in batch]
        metadatas = [
            {
                "company": c["company"],
                "year": c["year"],
                "filing_id": c["filing_id"],
                "chunk_method": c["chunk_method"],
                "section": c["section"]
            }
            for c in batch
        ]
        
        # Generate embeddings
        embeddings = model.encode(texts, show_progress_bar=False).tolist()
        
        # Add to ChromaDB
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
    
    print(f"Done! Collection '{collection_name}' has {collection.count()} documents.")
    return collection


if __name__ == "__main__":
    # Build vectorstore for fixed chunks
    print("=" * 50)
    print("Building vectorstore for FIXED chunks")
    print("=" * 50)
    build_vectorstore("data/chunks/fixed_chunks.json", "fixed_chunks")
    
    print()
    
    # Build vectorstore for section chunks
    print("=" * 50)
    print("Building vectorstore for SECTION chunks")
    print("=" * 50)
    build_vectorstore("data/chunks/section_chunks.json", "section_chunks")
    
    print("\nAll vectorstores built!")