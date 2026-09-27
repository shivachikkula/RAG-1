# LinkedIn post: RAG-1

## Main post (paste as-is)

I built 14 different RAG systems in one repo, so you don't have to guess which one you need. 🧵

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

My biggest takeaway: fix retrieval before you touch the prompt or the model. If the right chunk never reaches the LLM, no prompt can save the answer.

Swipe through the carousel for the architecture and a "which RAG, when?" cheat sheet ➡️

🔗 Code: github.com/shivachikkula/RAG-1

Which RAG pattern has worked best for you in production? 👇

#RAG #GenerativeAI #LLM #AIEngineering #Python #MachineLearning #Claude #OpenSource

---

## Shorter version (if you prefer brevity)

There isn't one "RAG". There are more than a dozen patterns, each fixing a different failure mode.

So I built 14 of them side by side in one Python repo: Naive, Hybrid, Advanced, Multi-Query, HyDE, Parent-Document, Contextual Retrieval, Conversational, Corrective, Self-RAG, Adaptive, Agentic, Graph and Citation RAG.

Every pattern is a runnable project on a shared core. It uses Claude for generation and runs fully offline for testing.

Takeaway: fix retrieval before you touch the prompt.

📑 Architecture + a "which RAG, when?" cheat sheet in the carousel
🔗 github.com/shivachikkula/RAG-1

#RAG #GenerativeAI #LLM #AIEngineering #Python

---

## First comment (post right after publishing, since links in comments often get more reach)

Repo: https://github.com/shivachikkula/RAG-1
Start with rag_types/01_naive_rag, then compare types:
python run.py naive hybrid graph -q "your question"

---

## Posting checklist

1. Make the GitHub repo **public** first, or the link will 404 for readers.
2. Upload `RAG-1-carousel.pdf` as a **document** post: Start a post → "+" → Add a document. It shows as a swipeable carousel. Give it a title like "14 ways to build RAG".
   - Or attach `slide1.png` as a single image post.
3. Paste the main post text.
4. Add the first comment with the link.
