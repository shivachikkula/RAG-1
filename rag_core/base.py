"""Common pipeline interface and the grounded-answer prompt."""

from __future__ import annotations

from dataclasses import dataclass, field

from .documents import Document, format_context
from .embeddings import Embedder, get_embedder
from .llm import BaseLLM, get_llm

ANSWER_SYSTEM = (
    "You answer questions using only the provided context. Cite sources inline as [n] "
    "using the context numbers. If the context does not contain the answer, say you don't know."
)

ANSWER_TEMPLATE = """<context>
{context}
</context>

<question>
{question}
</question>"""


@dataclass
class RAGResult:
    answer: str
    sources: list[Document] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def pretty(self) -> str:
        lines = [self.answer, "", "Sources:"]
        lines += [f"  [{i}] {d.id} ({d.source})" for i, d in enumerate(self.sources, 1)]
        for key, value in self.metadata.items():
            lines.append(f"{key}: {value}")
        return "\n".join(lines)


class BaseRAG:
    """Every RAG type implements ``index(docs)`` and ``query(question)``."""

    name = "base"

    def __init__(self, llm: BaseLLM | None = None, embedder: Embedder | None = None, top_k: int = 4):
        self.llm = llm or get_llm()
        self.embedder = embedder or get_embedder()
        self.top_k = top_k

    def index(self, docs: list[Document]) -> None:
        raise NotImplementedError

    def query(self, question: str) -> RAGResult:
        raise NotImplementedError

    def answer_from(self, question: str, docs: list[Document], **metadata) -> RAGResult:
        prompt = ANSWER_TEMPLATE.format(context=format_context(docs), question=question)
        answer = self.llm.generate(prompt, system=ANSWER_SYSTEM)
        return RAGResult(answer, docs, metadata)
