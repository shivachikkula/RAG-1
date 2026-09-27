"""Retrieval primitives: dense vector store, BM25 keyword index, and rank fusion."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from .documents import Document
from .embeddings import Embedder, tokenize


class VectorStore:
    """In-memory cosine-similarity vector store."""

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.docs: list[Document] = []
        self.matrix = np.zeros((0, embedder.dim), dtype=np.float32)

    def add(self, docs: list[Document], texts_to_embed: list[str] | None = None) -> None:
        """Add docs. ``texts_to_embed`` lets you embed something other than ``doc.text``
        (e.g. contextualised chunks or hypothetical questions)."""
        if not docs:
            return
        vectors = self.embedder.embed(texts_to_embed or [d.text for d in docs])
        self.docs.extend(docs)
        self.matrix = np.vstack([self.matrix, vectors])

    def search_by_vector(self, vector: np.ndarray, k: int = 4) -> list[tuple[Document, float]]:
        if not self.docs:
            return []
        scores = self.matrix @ vector
        top = np.argsort(-scores)[:k]
        return [(self.docs[i], float(scores[i])) for i in top]

    def search(self, query: str, k: int = 4) -> list[tuple[Document, float]]:
        return self.search_by_vector(self.embedder.embed_one(query), k)


class BM25Index:
    """Okapi BM25 keyword retrieval (sparse)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs: list[Document] = []
        self.term_freqs: list[Counter] = []
        self.doc_freq: Counter = Counter()
        self.avg_len = 0.0

    def add(self, docs: list[Document], texts_to_index: list[str] | None = None) -> None:
        for doc, text in zip(docs, texts_to_index or [d.text for d in docs]):
            tf = Counter(tokenize(text))
            self.docs.append(doc)
            self.term_freqs.append(tf)
            self.doc_freq.update(tf.keys())
        self.avg_len = sum(sum(tf.values()) for tf in self.term_freqs) / max(1, len(self.term_freqs))

    def search(self, query: str, k: int = 4) -> list[tuple[Document, float]]:
        n = len(self.docs)
        q_terms = tokenize(query)
        scores = []
        for doc, tf in zip(self.docs, self.term_freqs):
            length = sum(tf.values())
            score = 0.0
            for term in q_terms:
                if term not in tf:
                    continue
                idf = math.log(1 + (n - self.doc_freq[term] + 0.5) / (self.doc_freq[term] + 0.5))
                freq = tf[term]
                score += idf * freq * (self.k1 + 1) / (freq + self.k1 * (1 - self.b + self.b * length / self.avg_len))
            scores.append((doc, score))
        scores.sort(key=lambda pair: -pair[1])
        return [s for s in scores[:k] if s[1] > 0]


def reciprocal_rank_fusion(result_lists: list[list[tuple[Document, float]]], k: int = 60, top_n: int = 4):
    """Merge several ranked lists: score(d) = sum(1 / (k + rank))."""
    fused: dict[str, float] = {}
    by_id: dict[str, Document] = {}
    for results in result_lists:
        for rank, (doc, _) in enumerate(results):
            fused[doc.id] = fused.get(doc.id, 0.0) + 1.0 / (k + rank + 1)
            by_id[doc.id] = doc
    ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:top_n]
    return [(by_id[doc_id], score) for doc_id, score in ranked]
