# Agentic RAG

Claude drives retrieval itself through tool use. It decides what to search for, inspects results, reads full documents when a snippet is not enough, and searches again until it can answer.

## How it works

1. Tools: `search_knowledge_base(query, k)`, `list_documents()`, `read_document(source)` (all `strict: true`)
2. Manual agent loop: call Claude, run the requested tools, return all results in one message, repeat
3. Stop when Claude answers or the step limit is reached

## When to use it

Questions that need several hops across documents ("is my case covered?"), and exploratory research.

## Trade-offs

Variable latency and cost. Always set a step limit. The mock LLM emulates a single search step.

## Run it

`python rag_types/12_agentic_rag/pipeline.py "If a Cumulus C2 breaks after 20 months because it carried 150 kg, is it covered?"`

or `python run.py agentic -q "If a Cumulus C2 breaks after 20 months because it carried 150 kg, is it covered?"`

The code is in [`pipeline.py`](pipeline.py).
