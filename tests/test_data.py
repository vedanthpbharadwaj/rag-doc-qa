from ragqa.ingest import documents_from_beir, documents_from_directory


def test_load_beir(dataset):
    assert len(dataset.corpus) == 12
    assert len(dataset.queries) == 5
    assert set(dataset.test_queries) == {"1", "2", "3", "4"}  # query 5 has no qrels
    assert dataset.qrels["2"] == {"103": 1}


def test_documents_from_beir(dataset):
    docs = documents_from_beir(dataset.corpus, source="beir_mini")
    first = next(d for d in docs if d.metadata["doc_id"] == "101")
    assert first.page_content.startswith("Aspirin and cardiovascular events\n\n")
    assert first.metadata["title"] == "Aspirin and cardiovascular events"


def test_documents_from_directory(tmp_path):
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "a.md").write_text("# Policy\nRefunds within 30 days.")
    (tmp_path / "b.txt").write_text("Shipping takes 5 days.")
    (tmp_path / "skip.bin").write_bytes(b"\x00\x01")
    (tmp_path / "empty.txt").write_text("   ")
    docs = documents_from_directory(tmp_path)
    assert sorted(d.metadata["doc_id"] for d in docs) == ["b.txt", "notes/a.md"]


def _jsonl_to_parquet(path):
    import io
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows.append({"_id": str(row["_id"]), "title": row.get("title", ""), "text": row.get("text", "")})
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(rows), buf)
    return buf.getvalue()


def test_download_from_hugging_face_parquet(tmp_path, monkeypatch):
    import io

    from ragqa.data import beir

    fixture = beir.Path(__file__).parent / "fixtures" / "beir_mini"
    payloads = {
        "corpus-00000-of-00001.parquet": _jsonl_to_parquet(fixture / "corpus.jsonl"),
        "queries-00000-of-00001.parquet": _jsonl_to_parquet(fixture / "queries.jsonl"),
        "test.tsv": (fixture / "qrels" / "test.tsv").read_bytes(),
    }

    def fake_urlopen(url, timeout=None):
        if url.endswith(".zip"):
            raise AssertionError("BEIR server should not be needed when Hugging Face works")
        name = url.rsplit("/", 1)[-1]
        if name not in payloads:
            raise OSError(f"404 Not Found: {name}")
        return io.BytesIO(payloads[name])

    monkeypatch.setattr(beir.urllib.request, "urlopen", fake_urlopen)
    path = beir.download_beir("scifact", tmp_path)
    ds = beir.load_beir(path)
    assert len(ds.corpus) == 12 and len(ds.test_queries) == 4
    assert ds.qrels["2"] == {"103": 1}


def test_download_falls_back_to_zip(tmp_path, monkeypatch):
    import io
    import zipfile

    from ragqa.data import beir

    fixture = beir.Path(__file__).parent / "fixtures" / "beir_mini"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for rel in ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv"):
            zf.write(fixture / rel, f"scifact/{rel}")

    def fake_urlopen(url, timeout=None):
        if not url.endswith(".zip"):
            raise OSError("Hugging Face down")
        return io.BytesIO(buf.getvalue())

    monkeypatch.setattr(beir.urllib.request, "urlopen", fake_urlopen)
    ds = beir.load_beir(beir.download_beir("scifact", tmp_path))
    assert len(ds.queries) == 5
    assert not (tmp_path / "scifact.zip").exists()
