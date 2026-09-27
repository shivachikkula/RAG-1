# Conversational RAG

Keeps chat history. Each follow-up question ("what about its battery?") is condensed into a standalone query before retrieval, and the answer is generated with the history in view.

## How it works

1. If there is history, rewrite the question as a standalone question (structured output)
2. Retrieve with the standalone question
3. Answer with the conversation and context
4. Append the turn to memory, keeping the last N turns

## When to use it

Chatbots and assistants that handle multi-turn conversations.

## Trade-offs

History grows over time. Cap the number of turns, or summarise older ones.

## Run it

Interactive: `python rag_types/08_conversational_rag/pipeline.py`

The code is in [`pipeline.py`](pipeline.py).
