using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using RagCore;

namespace RagTypes;

/// <summary>
/// HyDE (Hypothetical Document Embeddings): an agent first writes a plausible answer without any
/// context; search uses that passage's embedding, because answer-to-answer similarity is often
/// stronger than question-to-answer similarity.
/// </summary>
public sealed class HydeRag(LlmProvider llm, IEmbeddingGenerator<string, Embedding<float>> embedder) : RagPipeline(llm, embedder)
{
    private VectorStore _store = null!;
    private ChatClientAgent _writer = null!;

    public override string Name => "hyde";
    public override string Summary => "Search with the embedding of a hypothetical answer.";

    public override async Task IndexAsync(IReadOnlyList<Document> docs, CancellationToken ct = default)
    {
        _store = new VectorStore(Embedder);
        await _store.AddAsync(Chunker.Sentences(docs), ct: ct);
        _writer = CreateAgent("hypothesis-writer",
            "Write a short, factual-sounding passage (3-4 sentences) that would answer the question, as it might appear in " +
            "product documentation. It is used only for search, so plausible details are fine.");
    }

    public override async Task<RagResult> QueryAsync(string question, CancellationToken ct = default)
    {
        var hypothetical = await AskAsync(_writer, $"<question>\n{question}\n</question>", ct: ct);

        // Average the question and hypothetical-answer vectors, so a wrong guess can't fully derail search.
        var q = await _store.EmbedAsync(question, ct);
        var h = await _store.EmbedAsync(hypothetical, ct);
        var combined = q.Zip(h, (a, b) => (a + b) / 2).ToArray();
        var norm = MathF.Sqrt(combined.Sum(x => x * x));
        if (norm > 0)
            for (var i = 0; i < combined.Length; i++) combined[i] /= norm;

        var hits = _store.SearchByVector(combined, TopK);
        return await AnswerFromAsync(question, hits.Select(x => x.Doc).ToList(),
            new Dictionary<string, object?> { ["hypothetical_document"] = hypothetical.Length > 200 ? hypothetical[..200] : hypothetical }, ct);
    }
}
