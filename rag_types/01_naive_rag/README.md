# Naive RAG

The baseline every other pattern improves on: split documents into fixed-size chunks, embed them, retrieve the top-k most similar chunks for the question, and put them in the prompt.

## How it works

1. Load and chunk documents (`chunk_fixed`, 500 chars)
2. Embed chunks into the `VectorStore`
3. Embed the question and take the top-k by cosine similarity
4. Generate a grounded answer with [n] citations

## When to use it

Prototypes, small and clean corpora, and as a benchmark for measuring the other patterns.

## Trade-offs

Chunks can cut facts in half. Retrieval can miss paraphrases and exact terms. Nothing checks whether the retrieved context is relevant.

## Run it

`python rag_types/01_naive_rag/pipeline.py "How long does the Stratus S1 battery last?"`

or `python run.py naive -q "How long does the Stratus S1 battery last?"`

The code is in [`pipeline.py`](pipeline.py).
