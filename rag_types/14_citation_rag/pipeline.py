"""Citation RAG: pass retrieved chunks to Claude as ``document`` content blocks with
citations enabled. Claude returns answer text whose spans carry verifiable citations
(the exact quoted source text + which document it came from), instead of relying on
the model to write [n] markers by itself."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rag_core import BaseRAG, BM25Index, RAGResult, VectorStore, chunk_sentences, reciprocal_rank_fusion  # noqa: E402


class CitationRAG(BaseRAG):
    """Hybrid retrieval + Claude's native document citations."""

    name = "citation"

    def index(self, docs):
        chunks = chunk_sentences(docs)
        self.store = VectorStore(self.embedder)
        self.store.add(chunks)
        self.bm25 = BM25Index()
        self.bm25.add(chunks)

    def query(self, question):
        fused = reciprocal_rank_fusion(
            [self.store.search(question, self.top_k * 2), self.bm25.search(question, self.top_k * 2)], top_n=self.top_k
        )
        docs = [doc for doc, _ in fused]
        if self.llm.is_mock:  # citations need the real API
            return self.answer_from(question, docs, citations=[])

        content = [
            {
                "type": "document",
                "source": {"type": "text", "media_type": "text/plain", "data": d.text},
                "title": d.id,
                "citations": {"enabled": True},
            }
            for d in docs
        ]
        content.append({"type": "text", "text": f"Answer using only these documents. If they don't contain the answer, say so.\n\n{question}"})
        response = self.llm.create(messages=[{"role": "user", "content": content}], max_tokens=4096)
        if response.stop_reason == "refusal":
            self.llm.text_of(response)  # raises LLMRefusalError

        answer_parts, citations = [], []
        for block in response.content:
            if block.type != "text":
                continue
            answer_parts.append(block.text)
            for cite in getattr(block, "citations", None) or []:
                citations.append({"doc": cite.document_title, "quote": cite.cited_text.strip()})
                answer_parts.append(f"[{len(citations)}]")
        return RAGResult("".join(answer_parts).strip(), docs, {"citations": citations})


if __name__ == "__main__":
    from rag_core.cli import run_cli

    run_cli(CitationRAG, "What response time does Premium Support guarantee?")
