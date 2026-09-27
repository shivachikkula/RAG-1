# Multi-Query RAG / RAG-Fusion

The LLM writes several reformulations of the question. Each one is retrieved separately and the result lists are fused with RRF.

## How it works

1. Generate N alternative queries (structured output)
2. Retrieve for the original question and each variant
3. Fuse all lists with RRF
4. Generate the answer

## When to use it

Multi-part questions and ambiguous wording, when recall matters more than latency.

## Trade-offs

One extra LLM call plus N retrievals per query.

## Run it

`python rag_types/04_multi_query_rag/pipeline.py "Who founded Nimbus and how much money did they raise?"`

or `python run.py multi_query -q "Who founded Nimbus and how much money did they raise?"`

The code is in [`pipeline.py`](pipeline.py).
