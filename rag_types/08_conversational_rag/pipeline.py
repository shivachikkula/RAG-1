"""Conversational RAG: keeps chat history, condenses each follow-up ("what about its
battery?") into a standalone search query before retrieving, and answers with the
history in view."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rag_core import ANSWER_SYSTEM, ANSWER_TEMPLATE, BaseRAG, RAGResult, VectorStore, chunk_sentences, format_context  # noqa: E402

CONDENSE_SCHEMA = {
    "type": "object",
    "properties": {"standalone_question": {"type": "string"}},
    "required": ["standalone_question"],
    "additionalProperties": False,
}


class ConversationalRAG(BaseRAG):
    """History-aware query condensation + memory."""

    name = "conversational"

    def __init__(self, *args, max_turns: int = 6, **kwargs):
        super().__init__(*args, **kwargs)
        self.history: list[tuple[str, str]] = []
        self.max_turns = max_turns

    def index(self, docs):
        self.store = VectorStore(self.embedder)
        self.store.add(chunk_sentences(docs))

    def _history_text(self) -> str:
        return "\n".join(f"User: {q}\nAssistant: {a}" for q, a in self.history[-self.max_turns:])

    def condense(self, question: str) -> str:
        if not self.history:
            return question
        prompt = (
            "Rewrite the latest user question as a standalone question that can be understood "
            "without the conversation, resolving pronouns and references.\n\n"
            f"<conversation>\n{self._history_text()}\n</conversation>\n\n<question>\n{question}\n</question>"
        )
        return self.llm.generate_json(prompt, CONDENSE_SCHEMA)["standalone_question"].strip() or question

    def query(self, question):
        standalone = self.condense(question)
        docs = [doc for doc, _ in self.store.search(standalone, k=self.top_k)]
        prompt = ANSWER_TEMPLATE.format(context=format_context(docs), question=question)
        if self.history:
            prompt = f"<conversation>\n{self._history_text()}\n</conversation>\n\n{prompt}"
        answer = self.llm.generate(prompt, system=ANSWER_SYSTEM)
        self.history.append((question, answer))
        return RAGResult(answer, docs, {"standalone_question": standalone, "turn": len(self.history)})

    def reset(self):
        self.history.clear()


if __name__ == "__main__":
    from rag_core import load_directory

    rag = ConversationalRAG()
    rag.index(load_directory())
    print(f"# conversational  (llm: {rag.llm.model}) - type 'exit' to quit, 'reset' to clear history")
    while True:
        try:
            question = input("\nYou: ").strip()
        except EOFError:
            break
        if question in ("exit", "quit"):
            break
        if question == "reset":
            rag.reset()
            continue
        if question:
            print(rag.query(question).pretty())
