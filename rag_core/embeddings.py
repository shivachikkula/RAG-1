"""Embedding backends.

Claude does not provide an embeddings endpoint, so embeddings are pluggable:

* ``HashingEmbedder`` - dependency-free (numpy only), deterministic, works offline.
  Good enough to demonstrate every RAG pattern; swap it for a real model in production.
* ``SentenceTransformerEmbedder`` - local neural embeddings (``pip install sentence-transformers``).
"""

from __future__ import annotations

import hashlib
import os
import re

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have how in is it its of on or that the this to was were what when "
    "where which who why will with does do did can".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


class Embedder:
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


class HashingEmbedder(Embedder):
    """Feature-hashed bag of unigrams + bigrams, L2-normalised."""

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def _bucket(self, token: str) -> tuple[int, float]:
        h = int.from_bytes(hashlib.md5(token.encode()).digest()[:8], "little")
        return h % self.dim, (1.0 if (h >> 63) & 1 else -1.0)

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = tokenize(text)
            features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
            for feat in features:
                idx, sign = self._bucket(feat)
                out[i, idx] += sign
            norm = np.linalg.norm(out[i])
            if norm:
                out[i] /= norm
        return out


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True), dtype=np.float32)


def get_embedder() -> Embedder:
    if os.environ.get("RAG_EMBEDDER", "hashing").lower() == "sentence-transformers":
        return SentenceTransformerEmbedder()
    return HashingEmbedder()
