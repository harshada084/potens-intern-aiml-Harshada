"""
src/vectorstore.py

A minimal, dependency-light vector store. Stores chunk embeddings + metadata
in a single JSON file on disk and does similarity search with plain numpy
cosine similarity. No compiled/native database engine involved -- avoids
binary compatibility issues with very new Python versions.
"""

import json
import os
import numpy as np

STORE_PATH = "vector_store.json"


class VectorStore:
    def __init__(self, path=STORE_PATH):
        self.path = path
        self.records = []  # each: {id, text, embedding, source, page, chunk_id}

    def add(self, id, text, embedding, metadata):
        self.records.append({
            "id": id,
            "text": text,
            "embedding": embedding,
            **metadata,
        })

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.records, f)

    def load(self):
        if not os.path.exists(self.path):
            self.records = []
            return
        with open(self.path, "r", encoding="utf-8") as f:
            self.records = json.load(f)

    def query(self, query_embedding, top_k=5):
        if not self.records:
            return []

        query_vec = np.array(query_embedding)
        query_norm = np.linalg.norm(query_vec)

        scored = []
        for r in self.records:
            vec = np.array(r["embedding"])
            denom = (np.linalg.norm(vec) * query_norm)
            similarity = float(np.dot(query_vec, vec) / denom) if denom > 0 else 0.0
            distance = 1 - similarity  # convert similarity to a "distance" (lower = closer), matching chroma-style usage
            scored.append((distance, r))

        scored.sort(key=lambda x: x[0])
        top = scored[:top_k]

        return [
            {
                "text": r["text"],
                "source": r["source"],
                "page": r["page"],
                "chunk_id": r["chunk_id"],
                "distance": dist,
            }
            for dist, r in top
        ]

    def clear(self):
        self.records = []
        if os.path.exists(self.path):
            os.remove(self.path)