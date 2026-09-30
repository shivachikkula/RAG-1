using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using RagCore;

namespace RagTypes;

/// <summary>
/// Multi-Query RAG (RAG-Fusion): an agent writes several versions of the question, each version is
/// searched, and the result lists are merged with Reciprocal Rank Fusion. Better recall.
/// </summary>
public sealed class MultiQueryRag(LlmProvider llm, IEmbeddingGenerator<string, Embedding<float>> embedder, int numQueries = 4) : RagPipeline(llm, embedder)
{
    public sealed record QueryList(List<string> Queries);

    private VectorStore _store = null!;
    private ChatClientAgent _expander = null!;

    public override string Name => "multi_query";
    public override string Summary => "LLM query expansion + RRF over per-query results.";

    public override async Task IndexAsync(IReadOnlyList<Document> docs, CancellationToken ct = default)
    {
        _store = new VectorStore(Embedder);
        await _store.AddAsync(Chunker.Sentences(docs), ct: ct);
        _expander = CreateAgent("query-expander",
            $"Write {numQueries} different search queries that together cover the information needed to answer the question. Vary vocabulary and angle; split multi-part questions.");
    }

    public override async Task<RagResult> QueryAsync(string question, CancellationToken ct = default)
    {
        var generated = (await AskForAsync<QueryList>(_expander, $"<question>\n{question}\n</question>", ct)).Queries ?? [];
        var queries = new[] { question }.Concat(generated.Where(q => !string.IsNullOrWhiteSpace(q))).Distinct().Take(numQueries + 1).ToList();

        var resultLists = new List<IReadOnlyList<Hit>>();
        foreach (var query in queries)
            resultLists.Add(await _store.SearchAsync(query, TopK, ct));

        var fused = RankFusion.Reciprocal(resultLists, TopK);
        return await AnswerFromAsync(question, fused.Select(h => h.Doc).ToList(), new Dictionary<string, object?> { ["queries"] = queries }, ct);
    }
}
