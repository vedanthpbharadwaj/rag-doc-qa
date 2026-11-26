"""Download and load BEIR-format datasets (default: SciFact).

SciFact has 5,183 scientific abstracts and 300 test claims, each with expert
relevance judgements (qrels). That gives us a real corpus to search and a
labelled test set to measure retrieval quality without any LLM calls.

Layout on disk (standard BEIR):
    <data_dir>/scifact/corpus.jsonl     {"_id", "title", "text"}
    <data_dir>/scifact/queries.jsonl    {"_id", "text"}
    <data_dir>/scifact/qrels/test.tsv   query-id  corpus-id  score
"""

from __future__ import annotations

import csv
import io
import json
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
# Hugging Face hosts the same data as Parquet (corpus, queries) plus a separate
# qrels repo. It is tried first because it is more reliable from CI runners.
HF_BASE = "https://huggingface.co/datasets/BeIR"
HF_PARQUET = {
    "corpus.jsonl": "{base}/{name}/resolve/main/corpus/corpus-00000-of-00001.parquet",
    "queries.jsonl": "{base}/{name}/resolve/main/queries/queries-00000-of-00001.parquet",
}
HF_QRELS = "{base}/{name}-qrels/resolve/main/{split}.tsv"


@dataclass
class BeirDataset:
    name: str
    corpus: dict[str, dict[str, str]] = field(default_factory=dict)
    queries: dict[str, str] = field(default_factory=dict)
    qrels: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def test_queries(self) -> dict[str, str]:
        """Queries that have at least one relevant document in the split."""
        return {qid: q for qid, q in self.queries.items() if qid in self.qrels}


def download_beir(name: str = "scifact", data_dir: str | Path = "data") -> Path:
    target = Path(data_dir) / name
    if (target / "corpus.jsonl").exists():
        return target
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    try:
        _download_hf(name, target)
    except Exception as exc:  # noqa: BLE001 - fall back to the original server on any failure
        print(f"Hugging Face download failed ({exc}); trying the BEIR server")
        shutil.rmtree(target, ignore_errors=True)
        _download_zip(name, Path(data_dir))
    return target


def _fetch(url: str) -> bytes:
    print(f"Downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as resp:
        return resp.read()


def _download_zip(name: str, data_dir: Path) -> None:
    zip_path = data_dir / f"{name}.zip"
    url = BEIR_URL.format(name=name)
    print(f"Downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as resp, open(zip_path, "wb") as fh:
        shutil.copyfileobj(resp, fh)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(data_dir)
    zip_path.unlink()


def _download_hf(name: str, target: Path) -> None:
    import pyarrow.parquet as pq  # installed via ragas -> datasets

    target.mkdir(parents=True, exist_ok=True)
    for filename, url in HF_PARQUET.items():
        table = pq.read_table(io.BytesIO(_fetch(url.format(base=HF_BASE, name=name))))
        with open(target / filename, "w", encoding="utf-8") as fh:
            for row in table.to_pylist():
                record = {
                    "_id": str(row["_id"]),
                    "title": row.get("title") or "",
                    "text": row.get("text") or "",
                }
                fh.write(json.dumps(record) + "\n")

    qrels_dir = target / "qrels"
    qrels_dir.mkdir(parents=True, exist_ok=True)
    for split in ("test", "train"):
        url = HF_QRELS.format(base=HF_BASE, name=name, split=split)
        try:
            (qrels_dir / f"{split}.tsv").write_bytes(_fetch(url))
        except Exception:  # noqa: BLE001 - only the test split is required
            if split == "test":
                raise


def load_beir(path: str | Path, split: str = "test") -> BeirDataset:
    path = Path(path)
    ds = BeirDataset(name=path.name)

    with open(path / "corpus.jsonl", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                ds.corpus[str(row["_id"])] = {"title": row.get("title", ""), "text": row.get("text", "")}

    with open(path / "queries.jsonl", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                ds.queries[str(row["_id"])] = row["text"]

    with open(path / "qrels" / f"{split}.tsv", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        for row in reader:
            if not row or row[0] == "query-id":
                continue
            qid, did, score = row[0], row[1], int(row[2])
            if score > 0:
                ds.qrels.setdefault(qid, {})[did] = score
    return ds
