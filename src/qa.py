"""
src/qa.py

Core /ask logic using the lightweight JSON vector store.
"""

import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
from vectorstore import VectorStore

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

EMBEDDING_MODEL = "gemini-embedding-001"
GENERATION_MODEL = "gemini-2.5-flash"
TOP_K = 5
DISTANCE_THRESHOLD = 0.5  # cosine-distance based; tune after seeing real scores

_store = VectorStore()
_store.load()


def embed_query(text):
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return result.embeddings[0].values


def retrieve(question, top_k=TOP_K):
    query_embedding = embed_query(question)
    return _store.query(query_embedding, top_k=top_k)


def build_prompt(question, chunks):
    context_blocks = "\n\n".join(
        f"[{c['chunk_id']} | source: {c['source']} | page {c['page']}]\n{c['text']}"
        for c in chunks
    )
    return f"""You are answering questions using ONLY the context provided below.

STRICT RULES:
- Only use information present in the context. Do not use outside knowledge.
- If the context does not contain enough information to answer, respond with exactly:
  "The documents do not cover this question."
- When you do answer, be concise and factual.
- Do not speculate or fill gaps with assumptions.

CONTEXT:
{context_blocks}

QUESTION:
{question}

ANSWER:"""


def ask(question):
    chunks = retrieve(question)

    if not chunks or min(c["distance"] for c in chunks) > DISTANCE_THRESHOLD:
        return {
            "answer": "The documents do not cover this question.",
            "citations": [],
            "no_answer_found": True,
        }

    prompt = build_prompt(question, chunks)
    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
    )
    answer_text = response.text.strip()

    no_answer = "do not cover" in answer_text.lower()

    citations = [] if no_answer else [
        {
            "source": c["source"],
            "chunk_ref": f"{c['chunk_id']} / page {c['page']}",
            "snippet": c["text"][:200],
        }
        for c in chunks[:3]
    ]

    return {
        "answer": answer_text,
        "citations": citations,
        "no_answer_found": no_answer,
    }


if __name__ == "__main__":
    q = input("Ask a question: ")
    result = ask(q)
    print("\nANSWER:", result["answer"])
    print("\nNO ANSWER FOUND:", result["no_answer_found"])
    print("\nCITATIONS:")
    for c in result["citations"]:
        print(f"  - {c['source']} ({c['chunk_ref']}): {c['snippet']}...")