using Microsoft.Extensions.AI;
using RagCore;

namespace RagTypes;

/// <summary>Naive RAG: fixed-size chunks, embed, take the top-k most similar, answer.</summary>
public sealed class NaiveRag(LlmProvider llm, IEmbeddingGenerator<string, Embedding<float>> embedder) : RagPipeline(llm, embedder)
{
    private VectorStore _store = null!;

    public override string Name => "naive";
    public override string Summary => "Fixed-size chunks, dense retrieval, single generation call.";

    public override async Task IndexAsync(IReadOnlyList<Document> docs, CancellationToken ct = default)
    {
        _store = new VectorStore(Embedder);
        await _store.AddAsync(Chunker.Fixed(docs, chunkSize: 500), ct: ct);
    }

    public override async Task<RagResult> QueryAsync(string question, CancellationToken ct = default)
    {
        var hits = await _store.SearchAsync(question, TopK, ct);
        return await AnswerFromAsync(question, hits.Select(h => h.Doc).ToList(),
            new Dictionary<string, object?> { ["scores"] = hits.Select(h => Math.Round(h.Score, 3)).ToList() }, ct);
    }
}
