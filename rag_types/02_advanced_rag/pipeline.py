"""Advanced RAG: adds pre-retrieval (query rewriting) and post-retrieval (LLM reranking)
optimisations around the naive pipeline, plus sentence-aware chunking with overlap."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rag_core import BaseRAG, VectorStore, chunk_sentences  # noqa: E402

REWRITE_SCHEMA = {
    "type": "object",
    "properties": {"search_query": {"type": "string"}},
    "required": ["search_query"],
    "additionalProperties": False,
}

RERANK_SCHEMA = {
    "type": "object",
    "properties": {
        "ranking": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"passage": {"type": "integer"}, "relevance": {"type": "integer"}},
                "required": ["passage", "relevance"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["ranking"],
    "additionalProperties": False,
}


class AdvancedRAG(BaseRAG):
    """Query rewrite -> over-retrieve -> LLM rerank -> answer."""

    name = "advanced"

    def __init__(self, *args, candidates: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.candidates = candidates

    def index(self, docs):
        self.store = VectorStore(self.embedder)
        self.store.add(chunk_sentences(docs, max_chars=400, overlap_sentences=1))

    def rewrite(self, question: str) -> str:
        prompt = (
            "Rewrite the question into a concise, keyword-rich search query for a document index. "
            f"Expand abbreviations and drop filler words.\n\n<question>\n{question}\n</question>"
        )
        return self.llm.generate_json(prompt, REWRITE_SCHEMA)["search_query"].strip() or question

    def rerank(self, question: str, docs: list) -> list:
        passages = "\n\n".join(f"<passage id=\"{i}\">\n{d.text}\n</passage>" for i, d in enumerate(docs))
        prompt = (
            "Score how useful each passage is for answering the question, from 0 (irrelevant) to 10 "
            "(directly answers it). Return one entry per passage.\n\n"
            f"{passages}\n\n<question>\n{question}\n</question>"
        )
        ranking = self.llm.generate_json(prompt, RERANK_SCHEMA)["ranking"]
        scores = {r["passage"]: r["relevance"] for r in ranking if 0 <= r["passage"] < len(docs)}
        if not scores:  # e.g. mock LLM: keep retrieval order
            return docs
        order = sorted(range(len(docs)), key=lambda i: -scores.get(i, 0))
        return [docs[i] for i in order if scores.get(i, 0) > 0] or docs

    def query(self, question):
        search_query = self.rewrite(question)
        candidates = [doc for doc, _ in self.store.search(search_query, k=self.candidates)]
        top = self.rerank(question, candidates)[: self.top_k]
        return self.answer_from(question, top, search_query=search_query)


if __name__ == "__main__":
    from rag_core.cli import run_cli

    run_cli(AdvancedRAG, "What's covered if my picking robot's battery dies after a year?")
