# Hybrid RAG

Runs sparse keyword search (BM25) and dense vector search side by side, then merges the two rankings with Reciprocal Rank Fusion (RRF).

## How it works

1. Index the same chunks in BM25 and in the vector store
2. Search both with the question
3. Fuse with RRF: `score = sum(1 / (60 + rank))`
4. Generate the answer

## When to use it

Corpora with exact identifiers such as product codes, error codes, standards numbers and names, where embeddings alone are weak. This is a good default for production.

## Trade-offs

You maintain two indexes. RRF ignores raw scores, which is usually a feature.

## Run it

`python rag_types/03_hybrid_rag/pipeline.py "Which ISO standard do the robots comply with?"`

or `python run.py hybrid -q "Which ISO standard do the robots comply with?"`

The code is in [`pipeline.py`](pipeline.py).
