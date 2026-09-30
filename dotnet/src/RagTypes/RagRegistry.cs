using Microsoft.Extensions.AI;
using RagCore;

namespace RagTypes;

/// <summary>All RAG types by short name, in reading order.</summary>
public static class RagRegistry
{
    public static readonly IReadOnlyDictionary<string, Func<LlmProvider, IEmbeddingGenerator<string, Embedding<float>>, RagPipeline>> All =
        new Dictionary<string, Func<LlmProvider, IEmbeddingGenerator<string, Embedding<float>>, RagPipeline>>
        {
            ["naive"] = (l, e) => new NaiveRag(l, e),
            ["advanced"] = (l, e) => new AdvancedRag(l, e),
            ["hybrid"] = (l, e) => new HybridRag(l, e),
            ["multi_query"] = (l, e) => new MultiQueryRag(l, e),
            ["hyde"] = (l, e) => new HydeRag(l, e),
            ["parent_document"] = (l, e) => new ParentDocumentRag(l, e),
            ["contextual_retrieval"] = (l, e) => new ContextualRetrievalRag(l, e),
            ["conversational"] = (l, e) => new ConversationalRag(l, e),
            ["corrective"] = (l, e) => new CorrectiveRag(l, e),
            ["self_rag"] = (l, e) => new SelfRag(l, e),
            ["adaptive"] = (l, e) => new AdaptiveRag(l, e),
            ["agentic"] = (l, e) => new AgenticRag(l, e),
            ["graph"] = (l, e) => new GraphRag(l, e),
            ["citation"] = (l, e) => new CitationRag(l, e),
        };

    public static RagPipeline Create(string name, LlmProvider llm, IEmbeddingGenerator<string, Embedding<float>>? embedder = null) =>
        All.TryGetValue(name, out var factory)
            ? factory(llm, embedder ?? new HashingEmbeddingGenerator())
            : throw new ArgumentException($"Unknown RAG type '{name}'. Known types: {string.Join(", ", All.Keys)}");
}
