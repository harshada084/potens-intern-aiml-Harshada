"""
src/main.py

FastAPI app exposing /ask (multilingual-aware), /contradict, /documents.
Run with (from inside src/):
  python -m uvicorn main:app --reload
Then test at http://127.0.0.1:8000/docs
"""

from fastapi import FastAPI
from pydantic import BaseModel
from translate import ask_multilingual
from contradict import contradict, list_available_documents

app = FastAPI(title="Document Q&A with Citations")


class AskRequest(BaseModel):
    question: str


class ContradictRequest(BaseModel):
    doc_a: str
    doc_b: str


@app.post("/ask")
def ask_endpoint(req: AskRequest):
    """
    Automatically detects the question's language. If not English,
    translates to English for retrieval/generation, then translates
    the answer back to the original language.
    """
    return ask_multilingual(req.question)


@app.post("/contradict")
def contradict_endpoint(req: ContradictRequest):
    return contradict(req.doc_a, req.doc_b)


@app.get("/documents")
def documents_endpoint():
    return {"documents": list_available_documents()}


@app.get("/")
def root():
    return {"status": "ok", "message": "Document Q&A API is running"}