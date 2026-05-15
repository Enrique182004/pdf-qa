import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from query import cosine_similarity, top_k_chunks, ask


def test_cosine_similarity_identical():
    v = [1.0, 0.0, 0.0]
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-6


def test_cosine_similarity_orthogonal():
    assert abs(cosine_similarity([1.0, 0.0], [0.0, 1.0])) < 1e-6


def test_top_k_chunks_returns_sorted():
    vectors = [
        {"text": "far chunk", "embedding": [0.0, 1.0]},
        {"text": "close chunk", "embedding": [1.0, 0.0]},
        {"text": "medium chunk", "embedding": [0.7, 0.7]},
    ]
    query_emb = [1.0, 0.0]
    results = top_k_chunks(query_emb, vectors, k=2)
    assert results[0] == "close chunk"
    assert len(results) == 2


def test_ask_calls_gpt(tmp_path):
    vectors_path = tmp_path / "vectors.json"
    vectors_path.write_text(json.dumps([
        {"text": "Python is a programming language.", "embedding": [1.0, 0.0]},
        {"text": "Unrelated content here.", "embedding": [0.0, 1.0]},
    ]))

    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[1.0, 0.0])]
    )
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Python is a programming language."))]
    )

    answer = ask("What is Python?", mock_client, vectors_path=str(vectors_path))
    assert "Python" in answer
    mock_client.chat.completions.create.assert_called_once()
