"""Document model, loaders and chunking strategies shared by every RAG type."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_docs"


@dataclass
class Document:
    """A unit of text plus metadata. Chunks are Documents too."""

    text: str
    metadata: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.metadata.get("id", "")

    @property
    def source(self) -> str:
        return self.metadata.get("source", "unknown")


def load_directory(path: str | Path = DEFAULT_DATA_DIR, patterns=("*.md", "*.txt")) -> list[Document]:
    """Load every text/markdown file in a directory as one Document."""
    path = Path(path)
    docs = []
    for pattern in patterns:
        for file in sorted(path.glob(pattern)):
            docs.append(Document(file.read_text(encoding="utf-8"), {"source": file.name, "id": file.stem}))
    return docs


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_fixed(docs: list[Document], chunk_size: int = 500, overlap: int = 0) -> list[Document]:
    """Naive fixed-size character chunking (the baseline strategy)."""
    chunks = []
    step = max(1, chunk_size - overlap)
    for doc in docs:
        for n, start in enumerate(range(0, len(doc.text), step)):
            piece = doc.text[start : start + chunk_size].strip()
            if piece:
                meta = {**doc.metadata, "id": f"{doc.id}#{n}", "parent_id": doc.id, "chunk_index": n}
                chunks.append(Document(piece, meta))
    return chunks


def chunk_sentences(docs: list[Document], max_chars: int = 400, overlap_sentences: int = 1) -> list[Document]:
    """Sentence-aware chunking: packs whole sentences up to max_chars, with sentence overlap."""
    chunks = []
    for doc in docs:
        sentences = split_sentences(doc.text)
        current: list[str] = []
        n = 0
        for sentence in sentences:
            if current and sum(len(s) + 1 for s in current) + len(sentence) > max_chars:
                chunks.append(_make_chunk(doc, current, n))
                n += 1
                current = current[-overlap_sentences:] if overlap_sentences else []
            current.append(sentence)
        if current:
            chunks.append(_make_chunk(doc, current, n))
    return chunks


def _make_chunk(doc: Document, sentences: list[str], n: int) -> Document:
    meta = {**doc.metadata, "id": f"{doc.id}#{n}", "parent_id": doc.id, "chunk_index": n}
    return Document(" ".join(sentences), meta)


def format_context(docs: list[Document]) -> str:
    """Render retrieved chunks as numbered, source-tagged context for a prompt."""
    return "\n\n".join(f"[{i}] (source: {d.source})\n{d.text}" for i, d in enumerate(docs, 1))
