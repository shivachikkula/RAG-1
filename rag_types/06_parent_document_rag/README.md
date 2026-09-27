# Parent-Document (Small-to-Big) RAG

Indexes small child chunks (sentences) so matching is precise, but gives the LLM the larger parent section each match came from.

## How it works

1. Split documents into parent sections (paragraphs)
2. Split parents into child chunks (sentences) and index only the children
3. Retrieve children, then map them to unique parents
4. Answer from the parent sections

## When to use it

When precise matching needs small chunks but good answers need surrounding context, for example policies and manuals.

## Trade-offs

Uses more prompt tokens per retrieved hit.

## Run it

`python rag_types/06_parent_document_rag/pipeline.py "What happens when a person gets close to a robot?"`

or `python run.py parent_document -q "What happens when a person gets close to a robot?"`

The code is in [`pipeline.py`](pipeline.py).
