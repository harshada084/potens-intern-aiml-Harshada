"""
src/ingest.py

Loads PDFs from data/, splits them into overlapping text chunks,
embeds each chunk with Gemini, and stores everything in vector_store.json
via the lightweight VectorStore (no chromadb, avoids native crashes).

Run with:  python src/ingest.py
"""

import os
import glob
import time
from pypdf import PdfReader
from google import genai
from google.genai import types
from google.genai import errors
from dotenv import load_dotenv
from vectorstore import VectorStore

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

DATA_DIR = "data"
EMBEDDING_MODEL = "gemini-embedding-001"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def extract_text_by_page(pdf_path):
    try:
        reader = PdfReader(pdf_path)
    except Exception as e:
        print(f"    ERROR: could not open {pdf_path}: {e}")
        return []

    pages = []
    print(f"    PDF has {len(reader.pages)} page(s) according to PyPDF")
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as e:
            print(f"    ERROR extracting text from page {i + 1}: {e}")
            text = ""
        if text.strip():
            pages.append((i + 1, text))
    print(f"    -> {len(pages)} page(s) with usable text")
    return pages


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) <= chunk_size:
            current += (" " if current else "") + para
        else:
            if current:
                chunks.append(current)
            overlap_text = current[-overlap:] if current else ""
            current = (overlap_text + " " + para).strip()

    if current:
        chunks.append(current)

    return chunks


def embed_text(text, max_retries=5):
    for attempt in range(max_retries):
        try:
            result = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=text,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
            )
            time.sleep(0.8)
            return result.embeddings[0].values
        except errors.ClientError as e:
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                wait_time = 35 + (attempt * 10)
                print(f"    Rate limit hit, waiting {wait_time}s before retrying (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                raise
    raise RuntimeError("Failed to embed text after multiple retries due to rate limiting.")


def main():
    pdf_files = glob.glob(os.path.join(DATA_DIR, "*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {DATA_DIR}/. Add your documents there first.")
        return

    store = VectorStore()
    store.clear()  # fresh start each run

    chunk_counter = 0

    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        print(f"Processing {filename}...")
        pages = extract_text_by_page(pdf_path)

        for page_num, page_text in pages:
            chunks = chunk_text(page_text)
            for chunk in chunks:
                chunk_id = f"chunk_{chunk_counter}"
                embedding = embed_text(chunk)

                store.add(
                    id=chunk_id,
                    text=chunk,
                    embedding=embedding,
                    metadata={"source": filename, "page": page_num, "chunk_id": chunk_id},
                )
                chunk_counter += 1
                print(f"    embedded chunk {chunk_counter} ({filename}, page {page_num})")

        print(f"  -> {filename} done, running total: {chunk_counter} chunks")

    store.save()
    print(f"\nDone. Ingested {chunk_counter} chunks from {len(pdf_files)} PDFs into vector_store.json")


if __name__ == "__main__":
    main()