from ingest import chunk_text


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
