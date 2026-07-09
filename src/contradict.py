"""
src/contradict.py

/contradict logic: takes two document filenames (source IDs), pulls all
their chunks, and asks the LLM to reason about whether they conflict on
any topic.
"""

import os
from google import genai
from dotenv import load_dotenv
from vectorstore import VectorStore

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

GENERATION_MODEL = "gemini-2.5-flash"

_store = VectorStore()
_store.load()


def get_full_text(doc_id):
    """Reconstructs a document's text from all its stored chunks, in page order."""
    matching = [r for r in _store.records if r["source"] == doc_id]
    matching.sort(key=lambda r: (r["page"], r["chunk_id"]))
    return "\n\n".join(r["text"] for r in matching)


def list_available_documents():
    """Returns the distinct source filenames currently in the vector store."""
    return sorted(set(r["source"] for r in _store.records))


def build_contradiction_prompt(doc_a_id, text_a, doc_b_id, text_b):
    # Guard against extremely long documents blowing up the prompt;
    # trim to a safe size for a 24-hour build (full-doc analysis is a stretch goal).
    max_chars = 6000
    text_a = text_a[:max_chars]
    text_b = text_b[:max_chars]

    return f"""You are comparing two documents to check whether they contradict each other on any topic.

DOCUMENT A ({doc_a_id}):
{text_a}

DOCUMENT B ({doc_b_id}):
{text_b}

TASK:
- Determine if Document A and Document B contradict each other on any specific topic or claim.
- If they do, identify the topic, quote or closely paraphrase the conflicting claims from each document, and explain the contradiction.
- If they do not contradict each other (even if they discuss different things or agree), say so clearly.

Respond in this exact format:
CONFLICT: [true or false]
TOPIC: [the topic being compared, or "N/A" if no conflict]
REASONING: [your explanation]
"""


def contradict(doc_a_id, doc_b_id):
    text_a = get_full_text(doc_a_id)
    text_b = get_full_text(doc_b_id)

    if not text_a:
        return {"error": f"No document found matching '{doc_a_id}'. Available: {list_available_documents()}"}
    if not text_b:
        return {"error": f"No document found matching '{doc_b_id}'. Available: {list_available_documents()}"}

    prompt = build_contradiction_prompt(doc_a_id, text_a, doc_b_id, text_b)
    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
    )
    raw = response.text.strip()

    # Simple parse of the structured response
    conflict = False
    topic = "N/A"
    reasoning = raw

    for line in raw.split("\n"):
        if line.strip().upper().startswith("CONFLICT:"):
            conflict = "true" in line.lower()
        elif line.strip().upper().startswith("TOPIC:"):
            topic = line.split(":", 1)[1].strip()
        elif line.strip().upper().startswith("REASONING:"):
            reasoning = line.split(":", 1)[1].strip()

    return {
        "doc_a": doc_a_id,
        "doc_b": doc_b_id,
        "conflict": conflict,
        "topic": topic,
        "reasoning": reasoning,
        "raw_response": raw,
    }


if __name__ == "__main__":
    docs = list_available_documents()
    print("Available documents:")
    for d in docs:
        print(f"  - {d}")

    a = input("\nDocument A filename: ").strip()
    b = input("Document B filename: ").strip()
    result = contradict(a, b)
    print("\n--- RESULT ---")
    for k, v in result.items():
        if k != "raw_response":
            print(f"{k}: {v}")