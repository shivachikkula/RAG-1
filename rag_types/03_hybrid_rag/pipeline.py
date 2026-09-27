"""Hybrid RAG: combine sparse keyword search (BM25) with dense vector search and merge
the two rankings with Reciprocal Rank Fusion. Catches exact terms (IDs, product codes)
that embeddings miss, and paraphrases that keywords miss."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rag_core import BaseRAG, BM25Index, VectorStore, chunk_sentences, reciprocal_rank_fusion  # noqa: E402


class HybridRAG(BaseRAG):
    """BM25 + dense retrieval fused with RRF."""

    name = "hybrid"

    def index(self, docs):
        chunks = chunk_sentences(docs)
        self.store = VectorStore(self.embedder)
        self.store.add(chunks)
        self.bm25 = BM25Index()
        self.bm25.add(chunks)

    def query(self, question):
        dense = self.store.search(question, k=self.top_k * 2)
        sparse = self.bm25.search(question, k=self.top_k * 2)
        fused = reciprocal_rank_fusion([dense, sparse], top_n=self.top_k)
        return self.answer_from(
            question,
            [doc for doc, _ in fused],
            dense_hits=[d.id for d, _ in dense[:3]],
            sparse_hits=[d.id for d, _ in sparse[:3]],
        )


if __name__ == "__main__":
    from rag_core.cli import run_cli

    run_cli(HybridRAG, "Which ISO standard do the robots comply with?")
