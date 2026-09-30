using System.ComponentModel;
using System.Text.Json;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using RagCore;

namespace RagTypes;

/// <summary>
/// Agentic RAG: instead of a fixed retrieve-then-answer pipeline, the agent is given search tools
/// and decides for itself what to search for, reads full documents when a snippet isn't enough,
/// and repeats until it can answer. Microsoft Agent Framework runs the tool-calling loop.
/// </summary>
public sealed class AgenticRag(LlmProvider llm, IEmbeddingGenerator<string, Embedding<float>> embedder) : RagPipeline(llm, embedder)
{
    private const string Instructions =
        "You are a research assistant for the Nimbus Robotics knowledge base. Use the tools to find evidence before " +
        "answering; search several times with different queries if needed. Answer only from what the tools return and " +
        "cite chunk ids like [products#1]. If the knowledge base does not contain the answer, say so.";

    private readonly Dictionary<string, Document> _documents = [];
    private readonly Dictionary<string, Document> _seen = [];  // chunks the agent looked at during one query
    private VectorStore _store = null!;
    private Bm25Index _bm25 = null!;
    private ChatClientAgent _agent = null!;

    public override string Name => "agentic";
    public override string Summary => "The agent drives retrieval itself through tool calls.";

    public override async Task IndexAsync(IReadOnlyList<Document> docs, CancellationToken ct = default)
    {
        _documents.Clear();
        foreach (var doc in docs) _documents[doc.Source] = doc;
        var chunks = Chunker.Sentences(docs);
        _store = new VectorStore(Embedder);
        await _store.AddAsync(chunks, ct: ct);
        _bm25 = new Bm25Index();
        _bm25.Add(chunks);

        // Plain C# methods become tools the model can call. The [Description]s tell the model what they do.
        _agent = CreateAgent("researcher", Instructions,
        [
            AIFunctionFactory.Create(SearchKnowledgeBaseAsync, "search_knowledge_base"),
            AIFunctionFactory.Create(ListDocuments, "list_documents"),
            AIFunctionFactory.Create(ReadDocument, "read_document"),
        ]);
    }

    [Description("Hybrid keyword + semantic search over the knowledge base. Returns the best matching chunks with ids and sources.")]
    private async Task<string> SearchKnowledgeBaseAsync(
        [Description("Search query")] string query,
        [Description("Number of chunks to return (1-8)")] int k = 4)
    {
        k = Math.Clamp(k, 1, 8);
        var hits = RankFusion.Reciprocal([await _store.SearchAsync(query, k * 2), _bm25.Search(query, k * 2)], k);
        foreach (var hit in hits) _seen[hit.Doc.Id] = hit.Doc;
        return JsonSerializer.Serialize(hits.Select(h => new { id = h.Doc.Id, source = h.Doc.Source, text = h.Doc.Text }));
    }

    [Description("List every document in the knowledge base with its title.")]
    private string ListDocuments() =>
        JsonSerializer.Serialize(_documents.Values.Select(d => new { source = d.Source, title = d.Title }));

    [Description("Read the full text of one document by its source file name (from list_documents).")]
    private string ReadDocument([Description("Source file name, e.g. products.md")] string source)
    {
        if (!_documents.TryGetValue(source, out var doc))
            return $"No document named '{source}'. Call list_documents to see the names.";
        _seen[doc.Id] = doc;
        return doc.Text;
    }

    public override async Task<RagResult> QueryAsync(string question, CancellationToken ct = default)
    {
        _seen.Clear();
        var response = CheckRefusal(await _agent.RunAsync(question, cancellationToken: ct));
        var toolCalls = response.Messages.SelectMany(m => m.Contents).OfType<FunctionCallContent>()
            .Select(c => $"{c.Name}({JsonSerializer.Serialize(c.Arguments)})").ToList();
        return new RagResult(response.Text.Trim(), _seen.Values.ToList(), new Dictionary<string, object?> { ["tool_calls"] = toolCalls });
    }
}
