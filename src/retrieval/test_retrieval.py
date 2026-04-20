import chromadb
from sentence_transformers import SentenceTransformer

# Load model and database
model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="data/vectorstore")

# Test queries
test_questions = [
    "What were Apple's main risk factors?",
    "What was Tesla's automotive revenue?",
    "What challenges did Microsoft mention in their business outlook?",
]

# Test on both collections
for collection_name in ["fixed_chunks", "section_chunks"]:
    collection = client.get_collection(name=collection_name)
    print(f"\n{'='*60}")
    print(f"Collection: {collection_name} ({collection.count()} chunks)")
    print(f"{'='*60}")
    
    for question in test_questions:
        # Generate query embedding
        query_embedding = model.encode(question).tolist()
        
        # Search
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )
        
        print(f"\nQ: {question}")
        print("-" * 40)
        
        for i in range(len(results["ids"][0])):
            doc = results["documents"][0][i]
            meta = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            
            print(f"  Result {i+1}:")
            print(f"    Company: {meta['company']}")
            print(f"    Section: {meta['section']}")
            print(f"    Distance: {distance:.4f}")
            print(f"    Preview: {doc[:150]}...")
            print()