"""Corrective RAG (CRAG): grade every retrieved chunk for relevance, keep only the
relevant ones, and when the local knowledge base is insufficient fall back to web
search (Claude's server-side ``web_search`` tool) instead of answering from noise."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rag_core import ANSWER_SYSTEM, BaseRAG, RAGResult, VectorStore, chunk_sentences  # noqa: E402

GRADE_SCHEMA = {
    "type": "object",
    "properties": {
        "grades": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"passage": {"type": "integer"}, "relevant": {"type": "boolean"}},
                "required": ["passage", "relevant"],
                "additionalProperties": False,
            },
        },
        "sufficient": {"type": "boolean", "description": "Do the relevant passages fully answer the question?"},
    },
    "required": ["grades", "sufficient"],
    "additionalProperties": False,
}


class CorrectiveRAG(BaseRAG):
    """Retrieve -> grade -> (filter | web-search fallback) -> answer."""

    name = "corrective"

    def __init__(self, *args, web_fallback: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.web_fallback = web_fallback and not self.llm.is_mock

    def index(self, docs):
        self.store = VectorStore(self.embedder)
        self.store.add(chunk_sentences(docs))

    def grade(self, question, docs):
        passages = "\n\n".join(f"<passage id=\"{i}\">\n{d.text}\n</passage>" for i, d in enumerate(docs))
        prompt = (
            "For each passage decide whether it contains information relevant to the question. "
            "Then decide whether the relevant passages together are sufficient to answer it.\n\n"
            f"{passages}\n\n<question>\n{question}\n</question>"
        )
        result = self.llm.generate_json(prompt, GRADE_SCHEMA)
        irrelevant = {g["passage"] for g in result["grades"] if not g["relevant"]}
        return [d for i, d in enumerate(docs) if i not in irrelevant], result["sufficient"]

    def web_answer(self, question, relevant_docs) -> str:
        local = "\n\n".join(d.text for d in relevant_docs) or "(none)"
        messages = [{
            "role": "user",
            "content": (
                f"The internal knowledge base only had this:\n<context>\n{local}\n</context>\n\n"
                f"Search the web to fill the gaps, then answer.\n<question>\n{question}\n</question>"
            ),
        }]
        tools = [{"type": "web_search_20260209", "name": "web_search", "max_uses": 3}]
        for _ in range(3):  # pause_turn means a long server-tool turn needs resuming
            response = self.llm.create(system=ANSWER_SYSTEM, messages=messages, tools=tools, max_tokens=8000)
            if response.stop_reason != "pause_turn":
                break
            messages.append({"role": "assistant", "content": response.content})
        return self.llm.text_of(response)

    def query(self, question):
        candidates = [doc for doc, _ in self.store.search(question, k=self.top_k * 2)]
        relevant, sufficient = self.grade(question, candidates)
        meta = {"retrieved": len(candidates), "kept": len(relevant), "sufficient": sufficient}
        if not sufficient and self.web_fallback:
            return RAGResult(self.web_answer(question, relevant), relevant, {**meta, "action": "web_search"})
        if not relevant:
            return RAGResult("I don't know - nothing relevant was found in the knowledge base.", [], meta)
        return self.answer_from(question, relevant[: self.top_k], **meta, action="answer_from_kb")


if __name__ == "__main__":
    from rag_core.cli import run_cli

    run_cli(CorrectiveRAG, "What does the ISO 3691-4 standard cover?")
