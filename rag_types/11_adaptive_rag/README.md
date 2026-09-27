# Adaptive RAG

A router classifies each question and chooses the cheapest strategy that can answer it: no retrieval, single-step retrieval, or multi-step decomposition into sub-questions whose answers are then combined.

## How it works

1. Route: `no_retrieval` / `single_step` / `multi_step` (structured output with an enum)
2. single_step: standard RAG
3. multi_step: decompose, answer each sub-question with RAG, then synthesise

## When to use it

Mixed traffic where most questions are simple and a few are complex or comparative.

## Trade-offs

A wrong routing decision leads to a bad strategy. Log the routing decisions and evaluate them.

## Run it

`python rag_types/11_adaptive_rag/pipeline.py "Compare the payload and battery life of the Stratus S1 and the Cumulus C2."`

or `python run.py adaptive -q "Compare the payload and battery life of the Stratus S1 and the Cumulus C2."`

The code is in [`pipeline.py`](pipeline.py).
