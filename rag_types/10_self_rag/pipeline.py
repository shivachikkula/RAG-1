"""Self-RAG (self-reflective RAG): the model decides whether retrieval is needed,
generates an answer, then critiques it - is every claim supported by the context, and
does it actually answer the question? If not, it retrieves more and regenerates."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rag_core import BaseRAG, RAGResult, VectorStore, chunk_sentences, format_context  # noqa: E402

RETRIEVE_SCHEMA = {
    "type": "object",
    "properties": {"needs_retrieval": {"type": "boolean"}},
    "required": ["needs_retrieval"],
    "additionalProperties": False,
}

CRITIQUE_SCHEMA = {
    "type": "object",
    "properties": {
        "grounded": {"type": "boolean", "description": "Every claim is supported by the context"},
        "answers_question": {"type": "boolean"},
        "unsupported_claims": {"type": "array", "items": {"type": "string"}},
        "better_search_query": {"type": "string"},
    },
    "required": ["grounded", "answers_question", "unsupported_claims", "better_search_query"],
    "additionalProperties": False,
}


class SelfRAG(BaseRAG):
    """Retrieve-on-demand + generate + self-critique loop."""

    name = "self_rag"

    def __init__(self, *args, max_iterations: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_iterations = max_iterations

    def index(self, docs):
        self.store = VectorStore(self.embedder)
        self.store.add(chunk_sentences(docs))

    def needs_retrieval(self, question) -> bool:
        prompt = (
            "Does answering this question require looking up facts in the company knowledge base "
            "(products, policies, people), or is it general knowledge / chit-chat?\n\n"
            f"<question>\n{question}\n</question>"
        )
        return self.llm.generate_json(prompt, RETRIEVE_SCHEMA)["needs_retrieval"]

    def critique(self, question, docs, answer) -> dict:
        prompt = (
            "Critique the answer strictly against the context.\n\n"
            f"<context>\n{format_context(docs)}\n</context>\n\n<question>\n{question}\n</question>\n\n"
            f"<answer>\n{answer}\n</answer>\n\nIf the answer is not grounded or incomplete, suggest a "
            "better search query to find the missing information."
        )
        return self.llm.generate_json(prompt, CRITIQUE_SCHEMA)

    def query(self, question):
        if not self.needs_retrieval(question):
            return RAGResult(self.llm.generate(f"<question>\n{question}\n</question>"), [], {"retrieval": False})

        search_query, docs, trace = question, [], []
        for iteration in range(1, self.max_iterations + 1):
            seen = {d.id for d in docs}
            docs += [d for d, _ in self.store.search(search_query, k=self.top_k) if d.id not in seen]
            result = self.answer_from(question, docs)
            verdict = self.critique(question, docs, result.answer)
            trace.append({"iteration": iteration, "query": search_query,
                          "grounded": verdict["grounded"], "answers": verdict["answers_question"]})
            if verdict["grounded"] and verdict["answers_question"]:
                break
            search_query = verdict["better_search_query"].strip() or question
        result.metadata = {"retrieval": True, "trace": trace}
        return result


if __name__ == "__main__":
    from rag_core.cli import run_cli

    run_cli(SelfRAG, "How often must operators renew their safety certification?")
