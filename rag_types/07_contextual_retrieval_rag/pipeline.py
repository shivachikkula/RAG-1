"""Contextual Retrieval: before indexing, have the LLM write 1-2 sentences situating each
chunk within its whole document ("This chunk is from Nimbus's warranty policy and
describes..."), then prepend that context before embedding *and* BM25 indexing.
Fixes chunks that are ambiguous on their own. The full document is sent as a cached
system prefix, so contextualising N chunks of one document pays for it roughly once."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rag_core import BaseRAG, BM25Index, VectorStore, chunk_sentences, reciprocal_rank_fusion  # noqa: E402

CONTEXT_PROMPT = """Here is a chunk from the document above:
<chunk>
{chunk}
</chunk>
Give a short succinct context (1-2 sentences) to situate this chunk within the overall document
for the purposes of improving search retrieval of the chunk. Answer only with the context."""


class ContextualRetrievalRAG(BaseRAG):
    """LLM-contextualised chunks + hybrid (dense + BM25) retrieval."""

    name = "contextual_retrieval"

    def contextualize(self, document, chunk) -> str:
        if self.llm.is_mock:  # offline fallback: use the document title as context
            return f"From '{document.text.splitlines()[0].lstrip('# ')}'."
        system = [{
            "type": "text",
            "text": f"<document>\n{document.text}\n</document>",
            "cache_control": {"type": "ephemeral"},
        }]
        return self.llm.generate(CONTEXT_PROMPT.format(chunk=chunk.text), system=system, max_tokens=256, effort="low")

    def index(self, docs):
        chunks, texts = [], []
        for doc in docs:
            for chunk in chunk_sentences([doc]):
                context = self.contextualize(doc, chunk)
                chunk.metadata["context"] = context
                chunks.append(chunk)
                texts.append(f"{context}\n\n{chunk.text}")
        self.store = VectorStore(self.embedder)
        self.store.add(chunks, texts_to_embed=texts)
        self.bm25 = BM25Index()
        self.bm25.add(chunks, texts_to_index=texts)

    def query(self, question):
        fused = reciprocal_rank_fusion(
            [self.store.search(question, k=self.top_k * 2), self.bm25.search(question, k=self.top_k * 2)],
            top_n=self.top_k,
        )
        docs = [doc for doc, _ in fused]
        return self.answer_from(question, docs, chunk_contexts=[d.metadata["context"][:80] for d in docs])


if __name__ == "__main__":
    from rag_core.cli import run_cli

    run_cli(ContextualRetrievalRAG, "How long is the warranty on batteries?")
