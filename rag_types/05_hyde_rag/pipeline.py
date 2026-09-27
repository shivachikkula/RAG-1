"""HyDE (Hypothetical Document Embeddings): ask the LLM to write a plausible answer
passage *without* context, embed that passage, and search with it. Answer-to-answer
similarity is often stronger than question-to-answer similarity."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rag_core import BaseRAG, VectorStore, chunk_sentences  # noqa: E402


class HyDERAG(BaseRAG):
    """Search with the embedding of a hypothetical answer."""

    name = "hyde"

    def index(self, docs):
        self.store = VectorStore(self.embedder)
        self.store.add(chunk_sentences(docs))

    def hypothesize(self, question: str) -> str:
        prompt = (
            "Write a short, factual-sounding passage (3-4 sentences) that would answer the question, "
            "as it might appear in product documentation. It is used only for search, so plausible "
            f"details are fine.\n\n<question>\n{question}\n</question>"
        )
        return self.llm.generate(prompt, max_tokens=512, effort="low")

    def query(self, question):
        hypothetical = self.hypothesize(question)
        # Average the question and hypothetical-doc vectors so a wrong guess can't fully derail search.
        vector = self.embedder.embed([question, hypothetical]).mean(axis=0)
        vector /= np.linalg.norm(vector) or 1.0
        hits = self.store.search_by_vector(vector, k=self.top_k)
        return self.answer_from(question, [doc for doc, _ in hits], hypothetical_document=hypothetical[:200])


if __name__ == "__main__":
    from rag_core.cli import run_cli

    run_cli(HyDERAG, "How quickly does the flagship robot recharge?")
