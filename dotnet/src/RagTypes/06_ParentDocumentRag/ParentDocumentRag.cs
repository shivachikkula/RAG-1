using Microsoft.Extensions.AI;
using RagCore;

namespace RagTypes;

/// <summary>
/// Parent-Document (small-to-big) RAG: search small child chunks (sentences) for precise matching,
/// but give the model the larger parent section (paragraph) each match came from.
/// </summary>
public sealed class ParentDocumentRag(LlmProvider llm, IEmbeddingGenerator<string, Embedding<float>> embedder) : RagPipeline(llm, embedder)
{
    private readonly Dictionary<string, Document> _parents = [];
    private VectorStore _store = null!;

    public override string Name => "parent_document";
    public override string Summary => "Retrieve on small chunks, return their parent sections.";

    public override async Task IndexAsync(IReadOnlyList<Document> docs, CancellationToken ct = default)
    {
        _parents.Clear();
        foreach (var doc in docs)
        {
            var paragraphs = doc.Text.Split("\n\n").Select(p => p.Trim()).Where(p => p.Length > 0).ToList();
            for (var n = 0; n < paragraphs.Count; n++)
            {
                var id = $"{doc.Id}/p{n}";
                _parents[id] = new Document { Id = id, Source = doc.Source, Text = paragraphs[n] };
            }
        }
        var children = Chunker.Sentences(_parents.Values, maxChars: 150, overlapSentences: 0);
        _store = new VectorStore(Embedder);
        await _store.AddAsync(children, ct: ct);
    }

    public override async Task<RagResult> QueryAsync(string question, CancellationToken ct = default)
    {
        var hits = await _store.SearchAsync(question, TopK * 3, ct);
        var parents = hits.Select(h => h.Doc.ParentId!).Distinct().Take(TopK).Select(id => _parents[id]).ToList();
        return await AnswerFromAsync(question, parents,
            new Dictionary<string, object?> { ["matched_children"] = hits.Take(3).Select(h => h.Doc.Text.Length > 60 ? h.Doc.Text[..60] : h.Doc.Text).ToList() }, ct);
    }
}
