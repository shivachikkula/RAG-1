# HyDE: Hypothetical Document Embeddings

The LLM first writes a hypothetical answer passage without any context. Search then uses the embedding of that passage, because answer-to-answer similarity is often stronger than question-to-answer similarity.

## How it works

1. LLM drafts a plausible answer passage
2. Embed the question and the draft, then average the two vectors
3. Vector search with the combined vector
4. Answer from the real retrieved chunks

## When to use it

Short or abstract questions whose wording is very different from the documents.

## Trade-offs

When the LLM's guess is badly wrong, it can steer search off course. Averaging with the question vector reduces this risk.

## Run it

`python rag_types/05_hyde_rag/pipeline.py "How quickly does the flagship robot recharge?"`

or `python run.py hyde -q "How quickly does the flagship robot recharge?"`

The code is in [`pipeline.py`](pipeline.py).
