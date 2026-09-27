# Self-RAG (Self-Reflective RAG)

The model decides whether retrieval is needed at all, generates an answer, then critiques it: is every claim supported by the context, and does it answer the question? If either check fails, it retrieves more with a better query and regenerates.

## How it works

1. Retrieval needed? If not, answer directly
2. Retrieve, then generate
3. Critique for groundedness and completeness, returning a better query (structured output)
4. Loop until both checks pass or the iteration cap is reached

## When to use it

High-stakes answers where unsupported claims are unacceptable.

## Trade-offs

Makes up to about 2 extra calls per iteration. Cap the iterations.

## Run it

`python rag_types/10_self_rag/pipeline.py "How often must operators renew their safety certification?"`

or `python run.py self_rag -q "How often must operators renew their safety certification?"`

The code is in [`pipeline.py`](pipeline.py).
