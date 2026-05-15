from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using only the "
    "provided context. If the context doesn't contain enough information, say so."
)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a, dtype=float), np.array(b, dtype=float)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def top_k_chunks(
    query_embedding: list[float],
    vectors: list[dict],
    k: int = 5,
) -> list[str]:
    scored = [
        (cosine_similarity(query_embedding, v["embedding"]), v["text"])
        for v in vectors
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [text for _, text in scored[:k]]


def ask(
    question: str,
    client: OpenAI,
    vectors_path: str = "vectors.json",
    k: int = 5,
) -> str:
    path = Path(vectors_path)
    if not path.exists():
        raise FileNotFoundError(
            f"{vectors_path} not found. Run `python main.py ingest <file.pdf>` first."
        )

    vectors = json.loads(path.read_text())

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[question],
    )
    query_embedding = response.data[0].embedding

    chunks = top_k_chunks(query_embedding, vectors, k=k)
    context = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(chunks))

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return completion.choices[0].message.content
