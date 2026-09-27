# Citation RAG

Passes retrieved chunks to Claude as `document` content blocks with `citations: {enabled: true}`. The answer comes back with verifiable citations: the exact quoted source text and the document it came from.

## How it works

1. Hybrid retrieval (BM25 + dense with RRF)
2. Send each chunk as a document block with citations enabled
3. Collect `cited_text` and `document_title` from each cited text block

## When to use it

Compliance, legal, support and anywhere users must be able to verify claims.

## Trade-offs

Citations cannot be combined with structured outputs (`output_config.format`) in the same request.

## Run it

`python rag_types/14_citation_rag/pipeline.py "What response time does Premium Support guarantee?"`

or `python run.py citation -q "What response time does Premium Support guarantee?"`

The code is in [`pipeline.py`](pipeline.py).
