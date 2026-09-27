# Contextual Retrieval

Before indexing, the LLM writes one or two sentences that place each chunk within its whole document. That context is prepended before both embedding and BM25 indexing. The full document is sent as a cached prompt prefix (`cache_control`), so contextualising many chunks from one document costs roughly one full read of it.

## How it works

1. Chunk each document
2. For each chunk, send the document (cached) and the chunk, and ask for a short situating context
3. Index `context + chunk` in the vector store and in BM25
4. Hybrid search with RRF, then answer

## When to use it

Chunks that are ambiguous on their own, such as "It has a payload limit of 120 kg" when the subject was named earlier.

## Trade-offs

Indexing makes one LLM call per chunk. Prompt caching keeps that affordable. Offline mode uses the document title as the context.

## Run it

`python rag_types/07_contextual_retrieval_rag/pipeline.py "How long is the warranty on batteries?"`

or `python run.py contextual_retrieval -q "How long is the warranty on batteries?"`

The code is in [`pipeline.py`](pipeline.py).
