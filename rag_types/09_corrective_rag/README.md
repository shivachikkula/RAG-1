# Corrective RAG (CRAG)

Grades every retrieved chunk for relevance and drops the irrelevant ones. When the knowledge base is not sufficient, it falls back to web search using Claude's server-side `web_search` tool instead of answering from noise.

## How it works

1. Retrieve candidates
2. LLM grades each passage as relevant or not, and judges whether together they are sufficient (structured output)
3. Sufficient: answer from the relevant passages only
4. Not sufficient: answer with web search plus whatever local context was relevant

## When to use it

Open-domain questions, and knowledge bases with coverage gaps where hallucination is costly.

## Trade-offs

Adds a grading call. The web fallback adds latency and cost, and must be allowed in your API org settings.

## Run it

`python rag_types/09_corrective_rag/pipeline.py "What does the ISO 3691-4 standard cover?"`

or `python run.py corrective -q "What does the ISO 3691-4 standard cover?"`

The code is in [`pipeline.py`](pipeline.py).
