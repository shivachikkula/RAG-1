# RAG-1: Retrieval-Augmented Generation, All Types

A Python project that implements **14 RAG architectures** side by side. Each one has its own folder, README and runnable pipeline, and all of them share a small core library. Generation uses Claude through the official Anthropic SDK. Every pipeline also runs **offline with no API key**, using a deterministic mock LLM.

```
RAG-1/
├── rag_core/                 # shared building blocks
│   ├── documents.py          #   Document model, loaders, fixed and sentence chunking
│   ├── embeddings.py         #   HashingEmbedder (offline) / SentenceTransformerEmbedder
│   ├── retrievers.py         #   VectorStore, BM25Index, reciprocal_rank_fusion
│   ├── llm.py                #   ClaudeLLM (Anthropic SDK) and MockLLM
│   ├── base.py               #   BaseRAG interface, RAGResult, grounded-answer prompt
│   └── cli.py
├── rag_types/                # one folder per RAG type
│   ├── 01_naive_rag/
│   ├── 02_advanced_rag/
│   ├── ...
│   └── 14_citation_rag/
├── data/sample_docs/         # small fictional knowledge base (Nimbus Robotics)
├── tests/                    # offline tests for every pipeline
└── run.py                    # run or compare any RAG type
```

## The RAG types

| # | Type | Core idea | Reach for it when |
|---|------|-----------|-------------------|
| 01 | [Naive](rag_types/01_naive_rag) | chunk, embed, top-k, answer | baseline, prototypes |
| 02 | [Advanced](rag_types/02_advanced_rag) | query rewrite plus LLM rerank | vague queries, noisy retrieval |
| 03 | [Hybrid](rag_types/03_hybrid_rag) | BM25 + dense, fused with RRF | exact terms, IDs, codes; the usual production default |
| 04 | [Multi-Query / RAG-Fusion](rag_types/04_multi_query_rag) | several query variants, fused | multi-part or ambiguous questions |
| 05 | [HyDE](rag_types/05_hyde_rag) | search with a hypothetical answer | short or abstract questions |
| 06 | [Parent-Document](rag_types/06_parent_document_rag) | match small chunks, return big ones | when answers need surrounding context |
| 07 | [Contextual Retrieval](rag_types/07_contextual_retrieval_rag) | LLM-written context per chunk (prompt-cached) | ambiguous chunks such as "it", "the plan" |
| 08 | [Conversational](rag_types/08_conversational_rag) | condense follow-ups, keep memory | chatbots |
| 09 | [Corrective (CRAG)](rag_types/09_corrective_rag) | grade chunks, fall back to web search | knowledge-base gaps, costly hallucinations |
| 10 | [Self-RAG](rag_types/10_self_rag) | self-critique and retry loop | high-stakes grounded answers |
| 11 | [Adaptive](rag_types/11_adaptive_rag) | route: none, single-step or multi-step | mixed simple and complex traffic |
| 12 | [Agentic](rag_types/12_agentic_rag) | Claude drives search through tool use | multi-hop and exploratory questions |
| 13 | [Graph](rag_types/13_graph_rag) | LLM-built knowledge graph traversal | relationship and multi-hop questions |
| 14 | [Citation](rag_types/14_citation_rag) | Claude's native document citations | verifiable, compliance-grade answers |

A rough progression: **01 → 02/03** (better retrieval) → **04–07** (better queries and chunks) → **08** (memory) → **09–11** (self-correction and routing) → **12–13** (agents and graphs) → **14** (verifiability).

## Quick start

```bash
pip install -r requirements.txt

# offline: no key needed, uses the mock LLM
python run.py --list
python run.py hybrid -q "Which ISO standard do the robots comply with?"

# compare several types on the same question
python run.py naive hybrid graph agentic -q "Who led the team that designed Cirrus?"

# run a single project directly
python rag_types/12_agentic_rag/pipeline.py "Is a Cumulus C2 overloaded to 150 kg covered by warranty?"

# live: generation with Claude
export ANTHROPIC_API_KEY=sk-ant-...
python run.py self_rag -q "How often must operators renew their safety certification?"

# point any pipeline at your own documents (.md / .txt)
python run.py hybrid --data /path/to/docs -q "..."

pytest   # offline tests for all 14 pipelines
```

## Running in GitHub Actions

| Workflow | Trigger | What it does |
|---|---|---|
| [`ci.yml`](.github/workflows/ci.yml) | every push and pull request | runs `pytest` and all 14 flows with the mock LLM on Python 3.11 and 3.12. Needs no secrets |
| [`run-rag.yml`](.github/workflows/run-rag.yml) | manual (**Actions → Run RAG flows → Run workflow**) | runs the RAG types and question you choose, in `mock` or `live` mode |

To use `live` mode, add your key once: **Settings → Secrets and variables → Actions → New repository secret**, with name `ANTHROPIC_API_KEY`. A live run fails immediately if the secret is missing, rather than falling back to the mock.

Each run posts a results table and every answer to the run's **Summary** page. The manual workflow also uploads `report.md` as a downloadable `rag-report` artifact.

The same thing works locally:

```bash
python run.py all --mock --report report.md
python run.py naive hybrid --live -q "Who founded Nimbus?"
```

## Configuration

| Env var | Default | Meaning |
|---|---|---|
| `ANTHROPIC_API_KEY` | unset | When set, `ClaudeLLM` is used. When unset, `MockLLM` is used |
| `RAG_MODEL` | `claude-opus-5` | Claude model for generation |
| `RAG_MOCK` | unset | `1` forces the mock LLM even when a key is set |
| `RAG_EMBEDDER` | `hashing` | `sentence-transformers` switches to neural embeddings (`pip install sentence-transformers`) |

### About the LLM layer

- All Claude calls go through `ClaudeLLM.create()`. It enables **server-side refusal fallbacks** (`fallbacks: "default"`, beta `server-side-fallback-2026-07-01`): if the model declines a request, the API re-runs it on a recommended fallback model in the same call. A refusal that survives the fallback raises `LLMRefusalError`.
- Graders, routers, rewriters and extractors use **structured outputs** (`output_config.format` with a JSON schema) at `effort: "low"`, so results always parse.
- Claude has no embeddings endpoint, so embeddings are pluggable. The built-in `HashingEmbedder` is dependency-free and deterministic, which makes it good for learning and tests. Swap in a neural embedder (or Voyage AI, OpenAI, etc.) for real workloads.
- `MockLLM` is extractive and not intelligent. It exists so every pipeline is runnable and testable offline. Pipelines that need real API features (tool use, web search, citations) degrade gracefully in mock mode.

## Adding a new RAG type

1. Create `rag_types/NN_my_rag/pipeline.py` with a class that subclasses `rag_core.BaseRAG` and implements `index(docs)` and `query(question) -> RAGResult`.
2. Register it in `rag_types/__init__.py` (`RAG_TYPES`).
3. Add a README. The parametrised test in `tests/test_pipelines.py` picks it up automatically.
