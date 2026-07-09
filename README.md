# Document Q&A with Citations

A RAG (Retrieval-Augmented Generation) system over 6 research papers on machine-learning-based credit card fraud detection. Built for the Potens AI/ML internship 24-hour assignment (Q1).

## Approach

This system ingests 6 fraud-detection research papers, chunks them, embeds each chunk with Gemini, and stores them in a lightweight custom JSON-based vector store (built after chromadb caused native crashes on Python 3.14 — see Known Limitations). The `/ask` endpoint retrieves the most relevant chunks and answers strictly from that context, refusing to answer when retrieval confidence is low. The `/contradict` endpoint compares two full documents and reasons about whether they conflict. A translation boundary enables multilingual queries: non-English questions are translated to English for retrieval, and answers are translated back. A Streamlit UI ties both features together for easy testing.

## Chunking Strategy

Documents are processed page-by-page using PyPDF text extraction. Each page is split into chunks of approximately 500 characters with 50 characters of overlap, breaking on paragraph boundaries where possible so sentences aren't cut mid-way. Every chunk stores its source filename, page number, and chunk ID, which is what powers accurate citations in `/ask` responses.

## Setup

1. Install dependencies:
   ```bash
   pip install fastapi uvicorn google-genai streamlit pypdf langdetect python-dotenv numpy
   ```
2. Get a free Gemini API key from https://aistudio.google.com/apikey
3. Create a `.env` file in the project root:
   ```
   GEMINI_API_KEY=your_key_here
   ```
4. Add your PDF documents to the `data/` folder
5. Run ingestion (embeds and stores all documents):
   ```bash
   python src/ingest.py
   ```
6. Run the API (from inside `src/`):
   ```bash
   cd src
   python -m uvicorn main:app --reload
   ```
   Interactive docs at `http://127.0.0.1:8000/docs`
7. Run the Streamlit UI (from the project root):
   ```bash
   python -m streamlit run app_streamlit.py
   ```

## Endpoints

- **POST /ask** — `{"question": "..."}` → `{answer, citations, no_answer_found, detected_language}`. Automatically detects and handles non-English questions.
- **POST /contradict** — `{"doc_a": "filename.pdf", "doc_b": "filename.pdf"}` → `{conflict, topic, reasoning}`
- **GET /documents** — lists all ingested document filenames

## Documents Used

Six arXiv papers on machine-learning-based credit card fraud detection, covering classical ML approaches, class-imbalance handling (SMOTE and variants), dataset shift, and deep learning methods — chosen to overlap enough for `/contradict` to have real substance to compare.

## Known Limitations

- **Vector store is a custom JSON + numpy implementation, not chromadb/FAISS/pgvector as suggested.** chromadb's compiled dependencies caused a native access-violation crash on Python 3.14 (a very recently released Python version with incomplete third-party binary support). Rather than downgrade the environment mid-assignment, I built a minimal, dependency-light vector store: chunk embeddings are stored in a single JSON file, and retrieval uses plain cosine similarity via numpy. This is correct and fully functional, but doesn't scale past a few thousand chunks (linear scan per query) — fine for this assignment's scope, not production-ready.
- `/contradict` sends each document's full text (truncated to ~6000 characters) to the LLM in a single call rather than doing exhaustive pairwise chunk comparison — sufficient for topic-level contradiction detection but could miss conflicts buried deep in very long documents.
- No confidence score or reranker (stretch goals, not attempted given the time available).
- Multilingual support translates at the query/answer boundary rather than embedding in multiple languages natively — acceptable per the assignment brief for a 24-hour build, but retrieval quality for non-English queries depends entirely on translation accuracy.
- Hallucination guard uses a fixed cosine-distance threshold (0.5) tuned by inspection on this document set; it isn't calibrated against a labeled eval set.

## Next Steps (if I had more time)

- Add a confidence score per answer with a human-in-the-loop gate below a threshold
- Add a reranker on top of vector retrieval
- Build a small eval set (10 Q&A pairs with ground truth) to measure retrieval@top-k
- Swap the JSON vector store for FAISS once running in a Python version with stable binary support
- More rigorous chunk-level contradiction detection for `/contradict` on longer documents

## AI Use Log

AI tools (Claude) were used throughout this project as follows:
- Scaffolding the initial FastAPI/ingestion/RAG project structure and file organization
- Debugging a series of environment issues: Windows terminal/PowerShell command syntax, git setup from scratch, API key format changes, and a native crash traced to chromadb's incompatibility with Python 3.14 — diagnosed by checking process exit codes and isolating the failure to the vector-store layer
- Migrating from the deprecated `google.generativeai` package to `google.genai`, and identifying correct current model names (`gemini-embedding-001`, `gemini-2.5-flash`) via a live model-list query against my API key
- Designing and writing the lightweight JSON+numpy vector store as a replacement for chromadb
- Writing the core retrieval, citation, hallucination-guard, contradiction, and translation logic
- Writing the Streamlit UI structure
- Drafting this README

All code was tested by me at each step (ingestion run against real documents, `/ask` and `/contradict` verified with real questions and outputs, multilingual flow tested in Hindi) before moving to the next piece. I do not have prior experience with git or terminal workflows, so AI assistance was especially used for git commands and Windows-specific debugging throughout.