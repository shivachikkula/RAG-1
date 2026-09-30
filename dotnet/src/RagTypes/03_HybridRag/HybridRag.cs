using Microsoft.Extensions.AI;
using RagCore;

namespace RagTypes;

/// <summary>
/// Hybrid RAG: keyword search (BM25) and vector search side by side, merged with Reciprocal Rank
/// Fusion. Keywords catch exact terms (codes, IDs); vectors catch paraphrases.
/// </summary>
public sealed class HybridRag(LlmProvider llm, IEmbeddingGenerator<string, Embedding<float>> embedder) : RagPipeline(llm, embedder)
{
    private VectorStore _store = null!;
    private Bm25Index _bm25 = null!;

    public override string Name => "hybrid";
    public override string Summary => "BM25 + dense retrieval fused with RRF.";

    public override async Task IndexAsync(IReadOnlyList<Document> docs, CancellationToken ct = default)
    {
        var chunks = Chunker.Sentences(docs);
        _store = new VectorStore(Embedder);
        await _store.AddAsync(chunks, ct: ct);
        _bm25 = new Bm25Index();
        _bm25.Add(chunks);
    }

    public override async Task<RagResult> QueryAsync(string question, CancellationToken ct = default)
    {
        var dense = await _store.SearchAsync(question, TopK * 2, ct);
        var sparse = _bm25.Search(question, TopK * 2);
        var fused = RankFusion.Reciprocal([dense, sparse], TopK);
        return await AnswerFromAsync(question, fused.Select(h => h.Doc).ToList(), new Dictionary<string, object?>
        {
            ["dense_hits"] = dense.Take(3).Select(h => h.Doc.Id).ToList(),
            ["sparse_hits"] = sparse.Take(3).Select(h => h.Doc.Id).ToList(),
        }, ct);
    }
}
