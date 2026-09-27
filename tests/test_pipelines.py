"""Offline smoke tests: every RAG type indexes the sample docs and answers with the mock LLM."""

import pytest

from rag_core import BM25Index, HashingEmbedder, MockLLM, VectorStore, chunk_sentences, load_directory, reciprocal_rank_fusion
from rag_types import RAG_TYPES, load_rag_class

DOCS = load_directory()


@pytest.mark.parametrize("name", list(RAG_TYPES))
def test_pipeline_answers(name):
    rag = load_rag_class(name)(llm=MockLLM(), embedder=HashingEmbedder())
    rag.index(DOCS)
    result = rag.query("How long does the Stratus S1 battery last on a single charge?")
    assert result.answer
    assert result.sources, f"{name} returned no sources"


def test_dense_retrieval_finds_battery_chunk():
    store = VectorStore(HashingEmbedder())
    store.add(chunk_sentences(DOCS))
    top, _ = store.search("Stratus S1 battery hours single charge", k=1)[0]
    assert "10 hours" in top.text


def test_bm25_exact_term():
    index = BM25Index()
    index.add(chunk_sentences(DOCS))
    top, _ = index.search("ISO 3691-4", k=1)[0]
    assert "3691" in top.text


def test_rrf_merges_lists():
    chunks = chunk_sentences(DOCS)
    fused = reciprocal_rank_fusion([[(chunks[0], 1.0), (chunks[1], 0.5)], [(chunks[1], 1.0)]], top_n=2)
    assert fused[0][0].id == chunks[1].id


def test_conversational_keeps_history():
    rag = load_rag_class("conversational")(llm=MockLLM(), embedder=HashingEmbedder())
    rag.index(DOCS)
    rag.query("Tell me about the Cumulus C2")
    result = rag.query("What is its battery life?")
    assert result.metadata["turn"] == 2


def test_graph_builds_edges_offline():
    rag = load_rag_class("graph")(llm=MockLLM(), embedder=HashingEmbedder())
    rag.index(DOCS)
    assert "Tomas Lindqvist" in rag.edges
