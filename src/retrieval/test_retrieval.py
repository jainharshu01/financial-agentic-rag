import chromadb
from sentence_transformers import SentenceTransformer

# ============================================================
# CONFIG
# ============================================================

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Load model and database
model = SentenceTransformer(EMBED_MODEL_NAME)
client = chromadb.PersistentClient(path="data/vectorstore")

# ============================================================
# TEST QUESTIONS
# ============================================================

test_questions = [
    "What were Apple's main risk factors?",
    "What was Tesla's automotive revenue?",
    "What challenges did Microsoft mention in their business outlook?",
]

# ============================================================
# TEST BOTH COLLECTIONS
# ============================================================

for collection_name in ["fixed_chunks", "section_chunks"]:

    collection = client.get_collection(name=collection_name)

    print(f"\n{'=' * 60}")
    print(f"Collection: {collection_name} ({collection.count()} chunks)")
    print(f"{'=' * 60}")

    for question in test_questions:

        # Generate query embedding (BGE format)
        query_embedding = model.encode(
            QUERY_PREFIX + question,
            normalize_embeddings=True
        ).tolist()

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

            similarity = round((1 - distance) * 100, 2)

            print(f"  Result {i+1}:")
            print(f"    Company: {meta.get('company')}")
            print(f"    Year: {meta.get('year')}")
            print(f"    Section: {meta.get('section')}")
            print(f"    Similarity: {similarity}%")
            print(f"    Distance: {distance:.4f}")
            print(f"    Preview: {doc[:200]}...")
            print()