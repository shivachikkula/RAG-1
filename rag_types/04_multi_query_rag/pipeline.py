"""Multi-Query RAG / RAG-Fusion: have the LLM generate several reformulations of the
question, retrieve for each, and fuse the result lists with Reciprocal Rank Fusion.
Improves recall when a single phrasing misses relevant chunks."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rag_core import BaseRAG, VectorStore, chunk_sentences, reciprocal_rank_fusion  # noqa: E402

QUERIES_SCHEMA = {
    "type": "object",
    "properties": {"queries": {"type": "array", "items": {"type": "string"}}},
    "required": ["queries"],
    "additionalProperties": False,
}


class MultiQueryRAG(BaseRAG):
    """LLM query expansion + RRF over per-query results."""

    name = "multi_query"

    def __init__(self, *args, num_queries: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_queries = num_queries

    def index(self, docs):
        self.store = VectorStore(self.embedder)
        self.store.add(chunk_sentences(docs))

    def expand(self, question: str) -> list[str]:
        prompt = (
            f"Write {self.num_queries} different search queries that together cover the information "
            "needed to answer the question. Vary vocabulary and angle; split multi-part questions.\n\n"
            f"<question>\n{question}\n</question>"
        )
        queries = self.llm.generate_json(prompt, QUERIES_SCHEMA)["queries"]
        return list(dict.fromkeys([question, *[q for q in queries if q.strip()]]))[: self.num_queries + 1]

    def query(self, question):
        queries = self.expand(question)
        result_lists = [self.store.search(q, k=self.top_k) for q in queries]
        fused = reciprocal_rank_fusion(result_lists, top_n=self.top_k)
        return self.answer_from(question, [doc for doc, _ in fused], queries=queries)


if __name__ == "__main__":
    from rag_core.cli import run_cli

    run_cli(MultiQueryRAG, "Who founded Nimbus and how much money did they raise?")
