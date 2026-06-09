"""
src/retrieval/baseline_rag.py   (generation fixes only — architecture intact)

Baseline static RAG. Still deliberately simple: single dense query, no
routing, no BM25/fusion, no rerank, no XBRL, no retry. That is the whole
point of the comparison.

WHY THESE CHANGES (and why they keep the comparison FAIR)
---------------------------------------------------------
A research comparison should isolate ONE variable: the agentic pipeline.
Previously the baseline AND the agentic pipeline both generated with
`llama-3.1-8b-instant`, but the project write-up assumed 70B — and the 8B
model's table-reading weakness was being attributed to "the baseline being
worse" rather than to the model. To make the measured Baseline-vs-Agentic
delta reflect the PIPELINE (retrieval/routing/rerank/XBRL/retry) and not
the generator, the baseline now uses the SAME generation model and the
SAME answer prompt as the agentic pipeline. Everything that defines the
baseline as "static RAG" (the retrieval) is unchanged.

CHANGES:
- Generation model -> llama-3.3-70b-versatile (was 8b-instant).
- Per-chunk char budget -> 1500 for tables / 900 for text (was flat 600).
- Prompt -> same extract-the-number prompt as agentic (minus XBRL lines).
"""

import os
import chromadb

from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

from src.agent.company_parser import extract_companies

load_dotenv()

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

GENERATION_MODEL = "llama-3.3-70b-versatile"
TABLE_CHAR_BUDGET = 1500
TEXT_CHAR_BUDGET = 900

embedding_model = SentenceTransformer(EMBED_MODEL_NAME)
client = chromadb.PersistentClient(path="data/vectorstore")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def _embed_query(question):
    return embedding_model.encode(
        QUERY_PREFIX + question, normalize_embeddings=True
    ).tolist()


def retrieve_chunks(question, company=None, year=None, section=None, top_k=5):
    collection = client.get_collection(name="section_chunks")
    query_embedding = _embed_query(question)

    if company:
        target_companies = [company]
    else:
        target_companies = extract_companies(question)

    filters = []
    if target_companies:
        if len(target_companies) == 1:
            filters.append({"company": target_companies[0]})
        else:
            filters.append({"$or": [{"company": c} for c in target_companies]})
    if year:
        filters.append({"year": year})
    if section:
        filters.append({"section": section})

    if len(filters) == 0:
        where_filter = None
    elif len(filters) == 1:
        where_filter = filters[0]
    else:
        where_filter = {"$and": filters}

    return collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_filter,
    )


def generate_answer(question, results):
    context_parts = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        is_table = meta.get("content_type") == "table"
        budget = TABLE_CHAR_BUDGET if is_table else TEXT_CHAR_BUDGET
        doc = results["documents"][0][i][:budget]
        context_parts.append(
            f"[Source {i+1}] Company: {meta['company']} | "
            f"Year: {meta['year']} | Section: {meta['section']} \n\n{doc}"
        )
    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are a financial document analyst answering questions about SEC 10-K filings.

Use ONLY the provided context below. Do NOT use outside knowledge.

HOW TO READ THE CONTEXT:
- Blocks wrapped in [TABLE] ... [/TABLE] are financial statement tables.
  When the answer is a number, look inside these tables and EXTRACT the
  matching row/value. Do not ignore tables in favour of prose.

VOCABULARY (treat these as the SAME line item):
- "revenue" = "net sales" = "total net sales" = "total revenues" = "total revenue"
- "net income" = "net earnings" = "profit (for the year)"
- "operating income" = "operating profit" = "income from operations"

RULES:
1. Cite every source you use as (Source X) — always include the parentheses.
2. For cross-company comparisons, state EACH company's figure and cite a
   source for each.
3. Give the specific number (with units) whenever the question asks for one
   and the number is present in any source.
4. Only answer "Insufficient evidence in the provided documents." if the
   requested fact is genuinely absent from EVERY source above — never as a
   hedge when a relevant number is present in a [TABLE].

Context:
{context}

Question:
{question}

Answer:"""

    response = groq_client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content


def baseline_answer(question, company=None, year=None, section=None, top_k=5):
    print("\n" + "=" * 60)
    print("BASELINE STATIC RAG")
    print("=" * 60)
    print(f"\nQuestion: {question}")
    print(f"Company: {company or '[auto-detect]'}")
    print(f"Year: {year}")
    print(f"Section: {section}")

    results = retrieve_chunks(
        question=question, company=company,
        year=year, section=section, top_k=top_k,
    )

    distances = results["distances"][0]
    if not distances:
        return {
            "answer": "No relevant documents found.",
            "results": results, "retry_used": False,
            "avg_similarity": 0, "top_similarity": 0, "retrieved_chunks": 0,
        }

    avg_distance = round(sum(distances) / len(distances), 4)
    avg_similarity = round((1 - avg_distance) * 100, 2)
    top_similarity = round((1 - min(distances)) * 100, 2)

    print("\nRetrieved Sources:\n")
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        dist = results["distances"][0][i]
        print(f"[{i+1}] {meta['company']} | Year: {meta['year']} | "
              f"{meta['section']} | Distance: {dist:.4f}")

    print("\nGenerating answer...\n")
    answer = generate_answer(question, results)
    print(answer)
    print("\n" + "=" * 60)

    return {
        "answer": answer, "results": results, "retry_used": False,
        "avg_similarity": avg_similarity, "top_similarity": top_similarity,
        "retrieved_chunks": len(distances),
    }


if __name__ == "__main__":
    baseline_answer(
        question="Compare Tesla's revenue with Apple's in 2024",
        company=None,
    )