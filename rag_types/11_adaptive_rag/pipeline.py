"""Adaptive RAG: a router classifies each question and picks the cheapest strategy that
can answer it - no retrieval, single-step retrieval, or multi-step decomposition
(split into sub-questions, retrieve + answer each, then synthesise)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rag_core import BaseRAG, RAGResult, VectorStore, chunk_sentences  # noqa: E402

ROUTE_SCHEMA = {
    "type": "object",
    "properties": {
        "strategy": {"type": "string", "enum": ["single_step", "multi_step", "no_retrieval"]},
        "reason": {"type": "string"},
    },
    "required": ["strategy", "reason"],
    "additionalProperties": False,
}

DECOMPOSE_SCHEMA = {
    "type": "object",
    "properties": {"sub_questions": {"type": "array", "items": {"type": "string"}}},
    "required": ["sub_questions"],
    "additionalProperties": False,
}


class AdaptiveRAG(BaseRAG):
    """Query-complexity router over three strategies."""

    name = "adaptive"

    def index(self, docs):
        self.store = VectorStore(self.embedder)
        self.store.add(chunk_sentences(docs))

    def route(self, question) -> dict:
        prompt = (
            "Choose a strategy for answering the question against a company knowledge base:\n"
            "- no_retrieval: general knowledge or chit-chat, no lookup needed\n"
            "- single_step: one lookup answers it\n"
            "- multi_step: needs several facts combined, comparisons, or chained reasoning\n\n"
            f"<question>\n{question}\n</question>"
        )
        return self.llm.generate_json(prompt, ROUTE_SCHEMA)

    def single_step(self, question, **meta) -> RAGResult:
        return self.answer_from(question, [d for d, _ in self.store.search(question, k=self.top_k)], **meta)

    def multi_step(self, question, **meta) -> RAGResult:
        prompt = f"Break the question into 2-4 simple, self-contained sub-questions.\n\n<question>\n{question}\n</question>"
        subs = self.llm.generate_json(prompt, DECOMPOSE_SCHEMA)["sub_questions"] or [question]
        notes, sources = [], []
        for sub in subs:
            partial = self.single_step(sub)
            notes.append(f"Q: {sub}\nA: {partial.answer}")
            sources += [d for d in partial.sources if d.id not in {s.id for s in sources}]
        synthesis = (
            "Using the answered sub-questions, write a complete answer to the original question.\n\n"
            "<context>\n" + "\n\n".join(notes) + f"\n</context>\n\n<question>\n{question}\n</question>"
        )
        return RAGResult(self.llm.generate(synthesis), sources, {**meta, "sub_questions": subs})

    def query(self, question):
        decision = self.route(question)
        strategy = decision["strategy"]
        if strategy == "no_retrieval":
            answer = self.llm.generate(f"<question>\n{question}\n</question>")
            return RAGResult(answer, [], {"strategy": strategy, "reason": decision["reason"]})
        if strategy == "multi_step":
            return self.multi_step(question, strategy=strategy)
        return self.single_step(question, strategy=strategy)


if __name__ == "__main__":
    from rag_core.cli import run_cli

    run_cli(AdaptiveRAG, "Compare the payload and battery life of the Stratus S1 and the Cumulus C2.")
