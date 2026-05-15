from unittest.mock import MagicMock, patch
import json
import tempfile
from pathlib import Path

from ingest import chunk_text, embed_chunks, ingest


def test_chunk_text_basic():
    text = "a" * 1200
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) == 3
    assert all(len(c) <= 500 for c in chunks)


def test_chunk_text_overlap():
    text = "abcdefghij"  # 10 chars
    chunks = chunk_text(text, chunk_size=6, overlap=2)
    # chunk 0: [0:6]  = "abcdef"
    # chunk 1: [4:10] = "efghij"
    assert chunks[0] == "abcdef"
    assert chunks[1] == "efghij"


def test_chunk_text_short_text():
    text = "hello"
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert chunks == ["hello"]


def test_chunk_text_empty():
    chunks = chunk_text("", chunk_size=500, overlap=50)
    assert chunks == []


def test_embed_chunks_calls_openai():
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.1, 0.2, 0.3])]
    )
    result = embed_chunks(["hello world"], mock_client)
    assert len(result) == 1
    assert result[0] == [0.1, 0.2, 0.3]
    mock_client.embeddings.create.assert_called_once_with(
        model="text-embedding-3-small",
        input=["hello world"],
    )


def test_ingest_creates_vectors_json(tmp_path):
    pdf_path = tmp_path / "test.pdf"

    fake_page = MagicMock()
    fake_page.extract_text.return_value = "This is test content for the PDF."

    mock_reader = MagicMock()
    mock_reader.pages = [fake_page]

    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.1, 0.2])]
    )

    out_path = tmp_path / "vectors.json"

    with patch("ingest.pypdf.PdfReader", return_value=mock_reader):
        ingest(str(pdf_path), mock_client, out_path=str(out_path))

    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert isinstance(data, list)
    assert data[0]["text"] == "This is test content for the PDF."
    assert data[0]["embedding"] == [0.1, 0.2]
