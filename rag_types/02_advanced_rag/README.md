# Advanced RAG

Adds optimisations before and after retrieval: the LLM rewrites the question into a better search query, more candidates than needed are retrieved, and an LLM reranker scores them before the best few reach the prompt.

## How it works

1. Sentence-aware chunking with one sentence of overlap
2. **Pre-retrieval:** rewrite the question into a keyword-rich query (structured output)
3. Over-retrieve 10 candidates
4. **Post-retrieval:** LLM reranks candidates 0-10 and keeps the top-k
5. Generate the answer

## When to use it

When naive RAG retrieves plausible but wrong chunks, or users write vague or chatty questions.

## Trade-offs

Two extra LLM calls per query. A cross-encoder model can replace the LLM reranker to cut cost.

## Run it

`python rag_types/02_advanced_rag/pipeline.py "What's covered if my picking robot's battery dies after a year?"`

or `python run.py advanced -q "What's covered if my picking robot's battery dies after a year?"`

The code is in [`pipeline.py`](pipeline.py).
