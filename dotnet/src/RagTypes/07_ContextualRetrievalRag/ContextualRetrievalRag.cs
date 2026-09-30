using Anthropic.Models.Beta.Messages;
using Microsoft.Extensions.AI;
using RagCore;

namespace RagTypes;

/// <summary>
/// Contextual Retrieval: before indexing, an agent writes 1-2 sentences placing each chunk in its
/// whole document ("This chunk is from the warranty policy and ..."). That context is prepended
/// before embedding and BM25 indexing. The whole document is sent as a cached system block, so
/// contextualising many chunks of one document costs roughly one full read of it.
/// </summary>
public sealed class ContextualRetrievalRag(LlmProvider llm, IEmbeddingGenerator<string, Embedding<float>> embedder) : RagPipeline(llm, embedder)
{
    private VectorStore _store = null!;
    private Bm25Index _bm25 = null!;

    public override string Name => "contextual_retrieval";
    public override string Summary => "LLM-contextualised chunks + hybrid (dense + BM25) retrieval.";

    private async Task<string> ContextualizeAsync(Document document, Document chunk, CancellationToken ct)
    {
        if (Llm.IsMock)  // offline fallback: use the document title as the context
            return $"From '{document.Title}'.";

        // The document is the cacheable prefix; only the chunk changes between calls.
        var options = Llm.NewChatOptions(system:
        [
            new BetaTextBlockParam { Text = $"<document>\n{document.Text}\n</document>", CacheControl = new BetaCacheControlEphemeral() },
        ]);
        var agent = CreateAgent("contextualizer", options: options);
        return await AskAsync(agent,
            $"Here is a chunk from the document above:\n<chunk>\n{chunk.Text}\n</chunk>\n" +
            "Give a short succinct context (1-2 sentences) to situate this chunk within the overall document " +
            "for the purposes of improving search retrieval of the chunk. Answer only with the context.", ct: ct);
    }

    public override async Task IndexAsync(IReadOnlyList<Document> docs, CancellationToken ct = default)
    {
        var chunks = new List<Document>();
        var texts = new List<string>();
        foreach (var doc in docs)
            foreach (var chunk in Chunker.Sentences([doc]))
            {
                var context = await ContextualizeAsync(doc, chunk, ct);
                chunk.Metadata["context"] = context;
                chunks.Add(chunk);
                texts.Add($"{context}\n\n{chunk.Text}");
            }
        _store = new VectorStore(Embedder);
        await _store.AddAsync(chunks, texts, ct);
        _bm25 = new Bm25Index();
        _bm25.Add(chunks, texts);
    }

    public override async Task<RagResult> QueryAsync(string question, CancellationToken ct = default)
    {
        var fused = RankFusion.Reciprocal([await _store.SearchAsync(question, TopK * 2, ct), _bm25.Search(question, TopK * 2)], TopK);
        var docs = fused.Select(h => h.Doc).ToList();
        return await AnswerFromAsync(question, docs,
            new Dictionary<string, object?> { ["chunk_contexts"] = docs.Select(d => d.Metadata["context"] is var c && c.Length > 80 ? c[..80] : c).ToList() }, ct);
    }
}
