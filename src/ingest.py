"""
src/ingest.py

Loads PDFs from data/, splits them into overlapping text chunks,
embeds each chunk with Gemini, and stores everything in a local
Chroma vector database (./chroma_db).

Run with:  python src/ingest.py
"""

import os
import glob
from pypdf import PdfReader
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

DATA_DIR = "data"
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "docs"

# --- Chunking config ---
# Fixed-size chunking with overlap, split on paragraph boundaries where possible.
CHUNK_SIZE = 500       # approx characters per chunk (kept simple; token-based is a stretch goal)
CHUNK_OVERLAP = 50     # characters of overlap between consecutive chunks


def extract_text_by_page(pdf_path):
    """Returns a list of (page_number, text) tuples for a PDF."""
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((i + 1, text))
    return pages


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Splits text into overlapping chunks, preferring paragraph breaks.
    Falls back to a hard character cut if a paragraph is too long.
    """
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) <= chunk_size:
            current += (" " if current else "") + para
        else:
            if current:
                chunks.append(current)
            # start new chunk, carrying over the overlap from the end of the last chunk
            overlap_text = current[-overlap:] if current else ""
            current = (overlap_text + " " + para).strip()

    if current:
        chunks.append(current)

    return chunks


def embed_text(text):
    """Embeds a single string using Gemini's embedding model."""
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=text,
        task_type="retrieval_document",
    )
    return result["embedding"]


def main():
    pdf_files = glob.glob(os.path.join(DATA_DIR, "*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {DATA_DIR}/. Add your documents there first.")
        return

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    # Fresh start each run so re-ingesting doesn't duplicate chunks
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    all_ids, all_embeddings, all_documents, all_metadatas = [], [], [], []
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

                all_ids.append(chunk_id)
                all_embeddings.append(embedding)
                all_documents.append(chunk)
                all_metadatas.append({
                    "source": filename,
                    "page": page_num,
                    "chunk_id": chunk_id,
                })
                chunk_counter += 1

    collection.add(
        ids=all_ids,
        embeddings=all_embeddings,
        documents=all_documents,
        metadatas=all_metadatas,
    )

    print(f"\nDone. Ingested {chunk_counter} chunks from {len(pdf_files)} PDFs into {CHROMA_PATH}/")


if __name__ == "__main__":
    main()