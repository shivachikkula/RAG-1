"""Agentic RAG: Claude drives retrieval itself through tool use. It decides what to
search for, inspects results, reads whole documents when a snippet isn't enough, and
iterates until it can answer - instead of a fixed retrieve-then-generate pipeline."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rag_core import BaseRAG, BM25Index, RAGResult, VectorStore, chunk_sentences, reciprocal_rank_fusion  # noqa: E402

SYSTEM = (
    "You are a research assistant for the Nimbus Robotics knowledge base. Use the tools to find "
    "evidence before answering; search several times with different queries if needed. Answer only "
    "from what the tools return and cite chunk ids like [products#1]. If the knowledge base does "
    "not contain the answer, say so."
)

TOOLS = [
    {
        "name": "search_knowledge_base",
        "description": "Hybrid keyword + semantic search over the knowledge base. Returns the top matching chunks with ids and sources.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "k": {"type": "integer", "description": "Number of chunks to return (1-8)"},
            },
            "required": ["query", "k"],
            "additionalProperties": False,
        },
    },
    {
        "name": "list_documents",
        "description": "List every document in the knowledge base with its title.",
        "strict": True,
        "input_schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
    },
    {
        "name": "read_document",
        "description": "Read the full text of one document by its source file name (from list_documents).",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"source": {"type": "string"}},
            "required": ["source"],
            "additionalProperties": False,
        },
    },
]


class AgenticRAG(BaseRAG):
    """Claude + retrieval tools in an agent loop."""

    name = "agentic"

    def __init__(self, *args, max_steps: int = 8, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_steps = max_steps

    def index(self, docs):
        self.documents = {d.source: d for d in docs}
        chunks = chunk_sentences(docs)
        self.store = VectorStore(self.embedder)
        self.store.add(chunks)
        self.bm25 = BM25Index()
        self.bm25.add(chunks)

    # ---- tools -------------------------------------------------------------
    def search_knowledge_base(self, query: str, k: int = 4):
        k = max(1, min(int(k), 8))
        fused = reciprocal_rank_fusion([self.store.search(query, k * 2), self.bm25.search(query, k * 2)], top_n=k)
        return [doc for doc, _ in fused]

    def run_tool(self, name: str, args: dict, seen: dict) -> str:
        if name == "search_knowledge_base":
            hits = self.search_knowledge_base(args["query"], args.get("k", 4))
            seen.update({d.id: d for d in hits})
            return json.dumps([{"id": d.id, "source": d.source, "text": d.text} for d in hits])
        if name == "list_documents":
            return json.dumps([{"source": s, "title": d.text.splitlines()[0].lstrip("# ")} for s, d in self.documents.items()])
        if name == "read_document":
            doc = self.documents.get(args["source"])
            if doc is None:
                raise KeyError(f"No document named {args['source']!r}")
            seen[doc.id] = doc
            return doc.text
        raise KeyError(f"Unknown tool {name}")

    # ---- agent loop ----------------------------------------------------------
    def query(self, question):
        if self.llm.is_mock:  # the mock can't call tools: emulate one search step
            docs = self.search_knowledge_base(question, self.top_k)
            return self.answer_from(question, docs, steps=1, tool_calls=["search_knowledge_base"])

        messages = [{"role": "user", "content": question}]
        seen, calls = {}, []
        for step in range(1, self.max_steps + 1):
            response = self.llm.create(system=SYSTEM, tools=TOOLS, messages=messages, max_tokens=16000)
            if response.stop_reason != "tool_use":
                return RAGResult(self.llm.text_of(response), list(seen.values()), {"steps": step, "tool_calls": calls})
            messages.append({"role": "assistant", "content": response.content})
            results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                calls.append(f"{block.name}({json.dumps(block.input)})")
                try:
                    results.append({"type": "tool_result", "tool_use_id": block.id,
                                    "content": self.run_tool(block.name, block.input, seen)})
                except (KeyError, ValueError, TypeError) as exc:
                    results.append({"type": "tool_result", "tool_use_id": block.id,
                                    "content": str(exc), "is_error": True})
            messages.append({"role": "user", "content": results})  # all results in ONE message
        return RAGResult("Stopped: step limit reached before an answer.", list(seen.values()), {"tool_calls": calls})


if __name__ == "__main__":
    from rag_core.cli import run_cli

    run_cli(AgenticRAG, "If a Cumulus C2 breaks after 20 months because it carried 150 kg, is it covered?")
