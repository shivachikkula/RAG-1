"""Naive RAG: chunk -> embed -> top-k similarity search -> stuff into prompt -> answer."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rag_core import BaseRAG, VectorStore, chunk_fixed  # noqa: E402


class NaiveRAG(BaseRAG):
    """Fixed-size chunks, dense retrieval, single generation call."""

    name = "naive"

    def index(self, docs):
        self.store = VectorStore(self.embedder)
        self.store.add(chunk_fixed(docs, chunk_size=500))

    def query(self, question):
        hits = self.store.search(question, k=self.top_k)
        return self.answer_from(question, [doc for doc, _ in hits], scores=[round(s, 3) for _, s in hits])


if __name__ == "__main__":
    from rag_core.cli import run_cli

    run_cli(NaiveRAG, "How long does the Stratus S1 battery last?")
