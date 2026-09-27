"""Graph RAG: the LLM extracts (entity, relation, entity) triples from every chunk to
build a knowledge graph. At query time, entities in the question are matched to graph
nodes, their neighbourhood is expanded, and the answer is generated from the relevant
triples plus the chunks they came from. Strong on multi-hop, relationship questions
("who designed the software the flagship robot uses?")."""

import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rag_core import BaseRAG, RAGResult, VectorStore, chunk_sentences, format_context, tokenize  # noqa: E402

TRIPLES_SCHEMA = {
    "type": "object",
    "properties": {
        "triples": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "relation": {"type": "string"},
                    "object": {"type": "string"},
                },
                "required": ["subject", "relation", "object"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["triples"],
    "additionalProperties": False,
}

_PROPER_NOUN = re.compile(r"\b(?:[A-Z][a-zA-Z0-9-]*(?:\s+[A-Z0-9][a-zA-Z0-9-]*)*)\b")
_SKIP = {"The", "It", "Its", "Each", "Every", "If", "All", "To", "Customers", "Operators", "Damage", "Batteries"}


class GraphRAG(BaseRAG):
    """LLM-built knowledge graph + neighbourhood expansion."""

    name = "graph"

    def __init__(self, *args, hops: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.hops = hops

    # ---- graph construction -------------------------------------------------
    def extract_triples(self, chunk) -> list[tuple[str, str, str]]:
        prompt = (
            "Extract factual (subject, relation, object) triples from the text. Use canonical, "
            "fully-named entities (e.g. 'Stratus S1', not 'it'). Use short snake_case relations.\n\n"
            f"<text>\n{chunk.text}\n</text>"
        )
        triples = [
            (t["subject"].strip(), t["relation"].strip(), t["object"].strip())
            for t in self.llm.generate_json(prompt, TRIPLES_SCHEMA)["triples"]
            if t["subject"].strip() and t["object"].strip()
        ]
        return triples or self._cooccurrence_triples(chunk.text)

    @staticmethod
    def _cooccurrence_triples(text):
        """Offline fallback: proper nouns appearing in the same sentence are linked."""
        triples = []
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            ents = list(dict.fromkeys(e for e in _PROPER_NOUN.findall(sentence) if e not in _SKIP and len(e) > 2))
            triples += [(a, "mentioned_with", b) for i, a in enumerate(ents) for b in ents[i + 1:]]
        return triples

    def index(self, docs):
        self.chunks = chunk_sentences(docs)
        self.edges = defaultdict(set)  # entity -> {(relation, other, chunk_id)}
        self.chunk_by_id = {c.id: c for c in self.chunks}
        for chunk in self.chunks:
            for s, r, o in self.extract_triples(chunk):
                self.edges[s].add((r, o, chunk.id))
                self.edges[o].add((f"inverse:{r}", s, chunk.id))
        self.store = VectorStore(self.embedder)  # used to seed entities when no name matches
        self.store.add(self.chunks)

    # ---- query -------------------------------------------------------------------
    def match_entities(self, question) -> list[str]:
        q_tokens = set(tokenize(question))
        scored = []
        for entity in self.edges:
            e_tokens = set(tokenize(entity))
            if e_tokens and e_tokens <= q_tokens:
                scored.append((len(e_tokens), entity))
        return [e for _, e in sorted(scored, reverse=True)]

    def query(self, question):
        seeds = self.match_entities(question)
        if not seeds:  # seed from entities mentioned in the best-matching chunks
            top_ids = {d.id for d, _ in self.store.search(question, k=2)}
            seeds = [e for e, rels in self.edges.items() if any(cid in top_ids for _, _, cid in rels)]
        frontier, visited, triples, chunk_ids = set(seeds), set(), set(), []
        for _ in range(self.hops + 1):
            next_frontier = set()
            for entity in frontier - visited:
                visited.add(entity)
                for rel, other, cid in self.edges.get(entity, ()):
                    if not rel.startswith("inverse:"):
                        triples.add((entity, rel, other))
                    else:
                        triples.add((other, rel.removeprefix("inverse:"), entity))
                    next_frontier.add(other)
                    if cid not in chunk_ids:
                        chunk_ids.append(cid)
            frontier = next_frontier
        docs = [self.chunk_by_id[c] for c in chunk_ids[: self.top_k * 2]]
        facts = "\n".join(f"- {s} --{r}--> {o}" for s, r, o in sorted(triples)[:60])
        prompt = (
            f"<context>\nKnowledge-graph facts:\n{facts}\n\nSource passages:\n{format_context(docs)}\n</context>\n\n"
            f"<question>\n{question}\n</question>"
        )
        answer = self.llm.generate(prompt, system="Answer using the graph facts and passages. Follow relationships across multiple hops when needed. Cite passages as [n].")
        return RAGResult(answer, docs, {"seed_entities": seeds[:5], "triples_used": len(triples)})


if __name__ == "__main__":
    from rag_core.cli import run_cli

    run_cli(GraphRAG, "Who led the team that designed the software the Stratus S1 uses?")
