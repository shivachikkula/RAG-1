# RAG-1 showcase post

Post text to use with the slides (`slide1-3.png` or `RAG-1-carousel.pdf`).

## Full version

I built 14 different RAG systems in one project, to learn which one fits which problem. 🧵

"Just add RAG" sounds simple, but there isn't one RAG. There are more than a dozen patterns, and each fixes a different failure mode. To really understand them, I implemented every major one side by side in Python.

What's inside 👇

🔵 Retrieval: how we search
• Naive: chunk → embed → top-k (the baseline)
• Hybrid: BM25 keywords + vector search, fused with Reciprocal Rank Fusion
• Parent-Document: match small chunks, return the full section

🟣 Query: ask better questions
• Advanced: LLM query rewriting + reranking
• Multi-Query / RAG-Fusion: several phrasings, results merged
• HyDE: search using a hypothetical answer
• Conversational: memory + follow-up condensing

🟢 Indexing: store knowledge smarter
• Contextual Retrieval: an LLM writes context for every chunk (with prompt caching)
• Graph RAG: a knowledge graph of entities and relations for multi-hop questions

🟠 Self-correcting: check before answering
• Corrective RAG: grade the retrieved chunks, fall back to web search
• Self-RAG: critique its own answer, retry if it isn't grounded
• Adaptive RAG: route each question to the cheapest strategy that works

🔴 Agents & trust
• Agentic RAG: the LLM drives the search through tool use
• Citation RAG: answers with verifiable, quoted sources

How it's built:
✅ One shared core (chunking, embeddings, vector store, BM25, rank fusion)
✅ Every pattern is its own runnable project with the same index() / query() interface
✅ Claude for generation: structured outputs, tool use, web search, citations
✅ Runs fully offline with a mock LLM, with 19 tests covering all 14 pipelines

Rule of thumb: fix retrieval before you touch the prompt or the model. If the right chunk never reaches the LLM, no prompt can save the answer.

The slides cover the architecture and a "which RAG, when?" cheat sheet ➡️

Which RAG pattern has worked best for you? 👇

#RAG #GenerativeAI #LLM #AIEngineering #Python #MachineLearning

---

## Short version

There isn't one "RAG". There are more than a dozen patterns, each fixing a different failure mode.

So I built 14 of them side by side in Python: Naive, Hybrid, Advanced, Multi-Query, HyDE, Parent-Document, Contextual Retrieval, Conversational, Corrective, Self-RAG, Adaptive, Agentic, Graph and Citation RAG.

Every pattern is a runnable project on a shared core. It uses Claude for generation and runs fully offline for testing.

Rule of thumb: fix retrieval before you touch the prompt.

📑 Architecture + a "which RAG, when?" cheat sheet in the slides.

#RAG #GenerativeAI #LLM #AIEngineering #Python
