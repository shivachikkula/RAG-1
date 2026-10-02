# RAG in .NET with Microsoft Agent Framework

The same 14 RAG types as the Python project, written in C# on **.NET 10** with
**[Microsoft Agent Framework](https://learn.microsoft.com/agent-framework/)** (`Microsoft.Agents.AI`).
Claude does the generation through the official Anthropic C# SDK. Everything also runs **offline
with no API key**, using a built-in mock model.

## 1. Install once

Install the **.NET 10 SDK** from https://dotnet.microsoft.com/download, then check it works:

```bash
dotnet --version     # should print 10.x
```

That's the only tool you need. There's no Node.js or Python here.

## 2. Run it

From this `dotnet/` folder:

```bash
dotnet run --project src/RagApp -- list                       # show the 14 RAG types
dotnet run --project src/RagApp -- naive                      # run one type (offline mock)
dotnet run --project src/RagApp -- hybrid graph -q "Who designed Cirrus?"
dotnet run --project src/RagApp -- all --report report.md     # run all 14, save a summary
dotnet run --project src/RagApp -- chat                       # chat with conversational RAG
dotnet run --project src/RagApp -- students                   # an agent using the Student MCP server
dotnet test                                                   # run all tests
```

The `--` separates `dotnet run`'s own options from the app's options. The first run downloads
packages and builds, so it takes a little longer.

**To use Claude instead of the mock**, set your API key first:

```bash
export ANTHROPIC_API_KEY=sk-ant-...                   # macOS / Linux
$env:ANTHROPIC_API_KEY = "sk-ant-..."                 # Windows PowerShell
dotnet run --project src/RagApp -- self_rag --live -q "How often must operators renew their safety certification?"
```

| Option | Meaning |
|---|---|
| `-q "..."` | the question to ask |
| `--mock` | always use the offline mock |
| `--live` | always use Claude; fails if `ANTHROPIC_API_KEY` isn't set |
| `--data <folder>` | use your own `.md` / `.txt` files instead of the sample docs |
| `--report <file>` | write a Markdown results table |

With neither `--mock` nor `--live`, it uses Claude if a key is set and the mock otherwise.
`RAG_MODEL` changes the Claude model (default `claude-opus-5`).

## 3. How the code is organised

```
dotnet/
├── RagDotNet.slnx               solution file (open this in Visual Studio / Rider / VS Code)
├── src/
│   ├── RagCore/                 shared building blocks
│   │   ├── Documents.cs         Document, loading files, chunking
│   │   ├── Embeddings.cs        offline embeddings (IEmbeddingGenerator)
│   │   ├── Retrievers.cs        VectorStore, Bm25Index, RankFusion
│   │   ├── LlmProvider.cs       Claude or mock, as an IChatClient
│   │   ├── MockChatClient.cs    the offline mock model
│   │   └── RagPipeline.cs       base class every RAG type extends
│   ├── RagTypes/                one folder per RAG type
│   │   ├── 01_NaiveRag/ ... 14_CitationRag/
│   │   └── RagRegistry.cs       name -> RAG type
│   ├── RagApp/Program.cs        the command-line app
│   └── StudentMcpServer/        MCP server: student database, files and a web API
└── tests/
    ├── RagTests/                xUnit tests for the RAG types
    └── StudentMcpServer.Tests/  xUnit tests for the MCP server
```

Every RAG type is a class with two methods:

```csharp
Task IndexAsync(IReadOnlyList<Document> docs);   // prepare the documents
Task<RagResult> QueryAsync(string question);      // answer a question
```

## 4. The Agent Framework ideas used

You only need four ideas to read any of the 14 types.

**1. `IChatClient`: the model.** The standard .NET interface for chat models. The Anthropic SDK
provides one for Claude, and `MockChatClient` is a second one. Because the pipelines only see
`IChatClient`, switching to another provider changes only `LlmProvider.cs`.

```csharp
IChatClient chat = new AnthropicClient().Beta.AsIChatClient("claude-opus-5");
```

**2. `ChatClientAgent`: a model with a job.** Each LLM step (rewriter, grader, critic, router,
answerer...) is an agent with its own instructions. `RagPipeline.CreateAgent` builds them:

```csharp
var grader = CreateAgent("grader", "Decide whether each passage is relevant to the question.");
string reply = await AskAsync(grader, prompt);
```

**3. Structured output: `RunAsync<T>`.** Ask for a C# record and get a typed object back.
Claude is constrained to a JSON schema generated from the record, so the reply always parses:

```csharp
public sealed record Critique(bool Grounded, bool AnswersQuestion, string BetterSearchQuery);
Critique c = await AskForAsync<Critique>(critic, prompt);   // uses agent.RunAsync<Critique>(...)
```

**4. Tools and sessions.**
- **Tools** (Agentic RAG): plain C# methods become tools via `AIFunctionFactory.Create`. The agent
  decides when to call them, and the framework runs the call-execute-repeat loop.
- **Sessions** (Conversational RAG): an `AgentSession` stores the chat history, so the agent
  remembers earlier turns without extra code.

## 5. The 14 types, and what each one uses

| # | Type | Idea | Agent Framework features |
|---|---|---|---|
| 01 | Naive | chunk → embed → top-k → answer | one answering agent |
| 02 | Advanced | rewrite the query, rerank results | 2 agents with `RunAsync<T>` |
| 03 | Hybrid | BM25 + vectors, fused with RRF | answering agent |
| 04 | Multi-Query | several phrasings, results fused | query-expander agent (`RunAsync<T>`) |
| 05 | HyDE | search with a hypothetical answer | writer agent |
| 06 | Parent-Document | match sentences, return paragraphs | answering agent |
| 07 | Contextual Retrieval | LLM-written context per chunk | contextualizer agent + Claude prompt caching |
| 08 | Conversational | condense follow-ups, remember the chat | `AgentSession` memory + condenser agent |
| 09 | Corrective (CRAG) | grade chunks, fall back to web search | grader agent + Claude web search tool |
| 10 | Self-RAG | critique the answer, retry | decider + critic agents in a loop |
| 11 | Adaptive | route by question complexity | router agent (enum output) + sub-agents |
| 12 | Agentic | the agent searches via tools | `AIFunctionFactory` tools, automatic tool loop |
| 13 | Graph | knowledge graph of facts | extractor agent (`RunAsync<T>` triples) |
| 14 | Citation | verifiable quoted sources | Claude document citations → `CitationAnnotation` |

The main README in the repository root explains each RAG pattern in more depth, and the
`rag_types/` folders there have a README per type.

## 6. Claude-specific extras

Some Claude features have no field in the generic `ChatOptions`. `LlmProvider.NewChatOptions`
supplies them through Agent Framework's `RawRepresentationFactory`, which is a template for
the underlying Anthropic request:

- **Refusal fallback** (all requests): if Claude declines, the API re-runs the request on
  Anthropic's recommended fallback model (`fallbacks: "default"`).
- **Prompt caching** (Contextual Retrieval): the full document is a cached system block.
- **Web search** (Corrective RAG): Claude's server-side `web_search` tool.
- **Citations** (Citation RAG): chunks are sent as document blocks with citations enabled.

`tests/RagTests/ClaudeRequestTests.cs` checks all four by capturing the real HTTP requests with a
fake handler, so it needs no API key.

## 7. Student MCP server

`src/StudentMcpServer` is a separate project: an **MCP server** that lets AI apps such as Claude
Desktop, Claude Code or an Agent Framework agent look up students. It reads from three sources:
a SQLite database (students, courses, grades), file storage (each student's documents) and an
external web API (Open Library textbook search). See
[its README](src/StudentMcpServer/README.md) for the tools and how to connect it to Claude.

## 8. Embeddings

Claude has no embeddings endpoint. `HashingEmbeddingGenerator` is a simple offline embedder
that's good enough for learning and tests. It implements the standard
`IEmbeddingGenerator<string, Embedding<float>>` interface, so you can swap in a real embedding
model (Azure OpenAI, Ollama, ONNX, Voyage AI...) by passing it to `RagRegistry.Create` or a
pipeline's constructor.
