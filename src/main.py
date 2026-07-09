"""
src/main.py

FastAPI app. Run with:
  uvicorn src.main:app --reload
Then test at http://127.0.0.1:8000/docs
"""

from fastapi import FastAPI
from pydantic import BaseModel
from src.qa import ask

app = FastAPI(title="Document Q&A with Citations")


class AskRequest(BaseModel):
    question: str


@app.post("/ask")
def ask_endpoint(req: AskRequest):
    return ask(req.question)


@app.get("/")
def root():
    return {"status": "ok", "message": "Document Q&A API is running"}