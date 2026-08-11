from retriever import (
    chunk_text,
    load_file,
    rank_chunks,
    retrieve,
)


def test_chunk_text_creates_overlapping_chunks():
    text = " ".join(f"word{i}" for i in range(1, 11))

    chunks = chunk_text(
        text,
        chunk_size=5,
        overlap=2
    )

    assert len(chunks) == 4

    assert chunks[0] == "word1 word2 word3 word4 word5"
    assert chunks[1] == "word4 word5 word6 word7 word8"
    assert chunks[2] == "word7 word8 word9 word10"
    assert chunks[3] == "word10"

def test_load_txt_file(tmp_path):
    file_path = tmp_path / "notes.txt"

    file_path.write_text(
        "A diode allows current to flow primarily in one direction.",
        encoding="utf-8"
    )

    chunks = load_file(
        str(file_path),
        "notes.txt"
    )

    assert len(chunks) > 0
    assert chunks[0]["source"] == "notes.txt"
    assert chunks[0]["type"] == "file"
    assert "diode" in chunks[0]["content"]


def test_rank_chunks_returns_most_relevant_first():
    chunks = [
        {
            "content": "A diode allows current to flow in one direction.",
            "source": "electronics.txt",
            "type": "file",
        },
        {
            "content": "Python is a programming language used for software development.",
            "source": "programming.txt",
            "type": "file",
        },
        {
            "content": "A capacitor stores electrical energy in an electric field.",
            "source": "electronics.txt",
            "type": "file",
        },
    ]

    ranked = rank_chunks(
        "How does a diode allow current to flow?",
        chunks,
        top_k=2
    )

    assert len(ranked) == 2
    assert ranked[0]["source"] == "electronics.txt"
    assert ranked[0]["score"] > 0
    assert ranked[0]["content"].startswith("A diode")


def test_retrieve_combines_sources_and_adds_source_ids(monkeypatch):
    file_chunks = [
        {
            "content": "FastAPI is a Python framework for building APIs.",
            "source": "notes.txt",
            "type": "file",
        }
    ]

    fake_web_results = [
        {
            "content": "FastAPI provides automatic API documentation.",
            "source": "FastAPI Documentation",
            "url": "https://example.com/fastapi",
            "type": "web",
        }
    ]

    monkeypatch.setattr(
        "retriever.web_search",
        lambda query, max_results=5: fake_web_results
    )

    results = retrieve(
        "What is FastAPI?",
        file_chunks,
        use_web=True,
        top_k=2
    )

    assert len(results) == 2

    assert all("source_id" in result for result in results)

    assert results[0]["source_id"] == 1
    assert results[1]["source_id"] == 2

    assert any(
        result["type"] == "web"
        for result in results
    )

    assert any(
        result["type"] == "file"
        for result in results
    )