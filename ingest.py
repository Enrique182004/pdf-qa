from __future__ import annotations
import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
import pypdf

load_dotenv()


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += chunk_size - overlap
    return chunks


def embed_chunks(chunks: list[str], client: OpenAI) -> list[list[float]]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=chunks,
    )
    return [item.embedding for item in response.data]


def ingest(pdf_path: str, client: OpenAI, out_path: str = "vectors.json") -> None:
    reader = pypdf.PdfReader(pdf_path)
    full_text = "\n".join(
        page.extract_text() or "" for page in reader.pages
    )
    chunks = chunk_text(full_text)
    if not chunks:
        raise ValueError(f"No text extracted from {pdf_path}")

    embeddings = embed_chunks(chunks, client)
    vectors = [{"text": c, "embedding": e} for c, e in zip(chunks, embeddings)]

    Path(out_path).write_text(json.dumps(vectors, indent=2))
    print(f"Ingested {len(vectors)} chunks → {out_path}")
