"""Parent-Document (small-to-big) RAG: index small child chunks for precise matching,
but hand the LLM the larger parent section each match came from, so answers keep
their surrounding context."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rag_core import BaseRAG, Document, VectorStore, chunk_sentences  # noqa: E402


class ParentDocumentRAG(BaseRAG):
    """Retrieve on small chunks, return their parent sections."""

    name = "parent_document"

    def index(self, docs):
        # Parents: paragraph-sized sections. Children: individual sentences.
        self.parents: dict[str, Document] = {}
        for doc in docs:
            for n, para in enumerate(p.strip() for p in doc.text.split("\n\n") if p.strip()):
                pid = f"{doc.id}/p{n}"
                self.parents[pid] = Document(para, {**doc.metadata, "id": pid})
        children = chunk_sentences(list(self.parents.values()), max_chars=150, overlap_sentences=0)
        self.store = VectorStore(self.embedder)
        self.store.add(children)

    def query(self, question):
        hits = self.store.search(question, k=self.top_k * 3)
        parent_ids = list(dict.fromkeys(doc.metadata["parent_id"] for doc, _ in hits))[: self.top_k]
        return self.answer_from(
            question,
            [self.parents[pid] for pid in parent_ids],
            matched_children=[doc.text[:60] for doc, _ in hits[:3]],
        )


if __name__ == "__main__":
    from rag_core.cli import run_cli

    run_cli(ParentDocumentRAG, "What happens when a person gets close to a robot?")
