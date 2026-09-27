# Graph RAG

The LLM extracts (subject, relation, object) triples from every chunk to build a knowledge graph. At query time, entities in the question are matched to graph nodes, their neighbourhood is expanded by N hops, and the answer uses both the triples and the source chunks they came from.

## How it works

1. Extract triples per chunk (structured output). Offline fallback: proper nouns in the same sentence are linked
2. Build an adjacency map with provenance (chunk ids)
3. Match question entities (or seed from the top vector hits) and expand the neighbourhood
4. Answer from the graph facts and passages

## When to use it

Relationship and multi-hop questions ("who designed the software the flagship robot uses?"), and entity-heavy domains.

## Trade-offs

Graph extraction is the costly part and happens at indexing time. Entity resolution (merging aliases) matters at scale. Consider a graph database such as Neo4j there.

## Run it

`python rag_types/13_graph_rag/pipeline.py "Who led the team that designed the software the Stratus S1 uses?"`

or `python run.py graph -q "Who led the team that designed the software the Stratus S1 uses?"`

The code is in [`pipeline.py`](pipeline.py).
