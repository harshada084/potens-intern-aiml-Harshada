"""
src/main.py

FastAPI app exposing /ask and /contradict.
Run with:
  uvicorn src.main:app --reload
Then test at http://127.0.0.1:8000/docs
"""

from fastapi import FastAPI
from pydantic import BaseModel
from qa import ask
from contradict import contradict, list_available_documents

app = FastAPI(title="Document Q&A with Citations")


class AskRequest(BaseModel):
    question: str


class ContradictRequest(BaseModel):
    doc_a: str
    doc_b: str


@app.post("/ask")
def ask_endpoint(req: AskRequest):
    return ask(req.question)


@app.post("/contradict")
def contradict_endpoint(req: ContradictRequest):
    return contradict(req.doc_a, req.doc_b)


@app.get("/documents")
def documents_endpoint():
    return {"documents": list_available_documents()}


@app.get("/")
def root():
    return {"status": "ok", "message": "Document Q&A API is running"}