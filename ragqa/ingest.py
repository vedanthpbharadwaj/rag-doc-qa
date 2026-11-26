"""Build the search index.

    python -m ragqa.ingest scifact                 # 5,183 SciFact abstracts (default corpus)
    python -m ragqa.ingest dir ./my_docs           # your own .pdf / .md / .txt files
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from langchain_core.documents import Document

from .config import get_settings
from .data.beir import download_beir, load_beir
from .embeddings import get_embeddings
from .index import DocumentIndex, chunk_documents

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst"}


def documents_from_beir(corpus: dict[str, dict[str, str]], source: str) -> list[Document]:
    docs = []
    for doc_id, row in corpus.items():
        title = row.get("title", "").strip()
        text = row.get("text", "").strip()
        content = f"{title}\n\n{text}" if title else text
        if content:
            docs.append(Document(page_content=content, metadata={"doc_id": doc_id, "title": title, "source": source}))
    return docs


def documents_from_directory(root: str | Path) -> list[Document]:
    root = Path(root)
    docs: list[Document] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(root))
        suffix = path.suffix.lower()
        if suffix in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                docs.append(Document(page_content=text, metadata={"doc_id": rel, "title": path.stem, "source": rel}))
        elif suffix == ".pdf":
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            for page_no, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    docs.append(Document(
                        page_content=text,
                        metadata={"doc_id": f"{rel}#p{page_no}", "title": path.stem, "source": rel, "page": page_no},
                    ))
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--index-dir", help="Override INDEX_DIR")
    sub = parser.add_subparsers(dest="source", required=True)
    b = sub.add_parser("scifact", parents=[common], help="Index a BEIR corpus (default SciFact)")
    b.add_argument("--dataset", default="scifact", help="Any BEIR dataset name")
    d = sub.add_parser("dir", parents=[common], help="Index a directory of .pdf/.md/.txt files")
    d.add_argument("path")
    args = parser.parse_args()

    settings = get_settings()
    started = time.time()
    if args.source == "scifact":
        ds = load_beir(download_beir(args.dataset, settings.data_dir))
        docs = documents_from_beir(ds.corpus, source=args.dataset)
    else:
        docs = documents_from_directory(args.path)
    print(f"Loaded {len(docs):,} documents")

    chunks = chunk_documents(docs, settings.chunk_size, settings.chunk_overlap)
    print(f"Split into {len(chunks):,} chunks (size {settings.chunk_size}, overlap {settings.chunk_overlap})")

    embeddings = get_embeddings(settings)
    index = DocumentIndex.build(chunks, embeddings, settings.embedding_model)
    out = args.index_dir or settings.index_dir
    index.save(out)
    print(f"Index saved to {out} in {time.time() - started:.0f}s: {index.manifest}")


if __name__ == "__main__":
    main()
