using System.Text;
using Anthropic.Models.Beta.Messages;
using Microsoft.Extensions.AI;
using RagCore;

namespace RagTypes;

/// <summary>
/// Citation RAG: retrieved chunks are sent to Claude as document blocks with citations turned on.
/// Claude's answer comes back with verifiable citations (the exact quoted source text and which
/// document it came from), exposed by the framework as <see cref="CitationAnnotation"/>s.
/// </summary>
public sealed class CitationRag(LlmProvider llm, IEmbeddingGenerator<string, Embedding<float>> embedder) : RagPipeline(llm, embedder)
{
    private VectorStore _store = null!;
    private Bm25Index _bm25 = null!;

    public override string Name => "citation";
    public override string Summary => "Hybrid retrieval + Claude's native document citations.";

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
        var fused = RankFusion.Reciprocal([await _store.SearchAsync(question, TopK * 2, ct), _bm25.Search(question, TopK * 2)], TopK);
        var docs = fused.Select(h => h.Doc).ToList();
        if (Llm.IsMock)  // citations need the real API
            return await AnswerFromAsync(question, docs, new Dictionary<string, object?> { ["citations"] = new List<string>() }, ct);

        // Document blocks with citations enabled go first in the request; the question follows.
        var documentsMessage = new BetaMessageParam
        {
            Role = Role.User,
            Content = docs.Select(d => (BetaContentBlockParam)new BetaRequestDocumentBlock
            {
                Source = new BetaPlainTextSource { Data = d.Text },
                Title = d.Id,
                Citations = new BetaCitationsConfigParam { Enabled = true },
            }).ToList(),
        };
        var agent = CreateAgent("citing-answerer",
            "Answer using only the provided documents. If they don't contain the answer, say so.",
            options: Llm.NewChatOptions(leadingMessages: [documentsMessage]));
        var response = CheckRefusal(await agent.RunAsync(question, cancellationToken: ct));

        var answer = new StringBuilder();
        var citations = new List<string>();
        foreach (var text in response.Messages.SelectMany(m => m.Contents).OfType<TextContent>())
        {
            answer.Append(text.Text);
            foreach (var cite in (text.Annotations ?? []).OfType<CitationAnnotation>())
            {
                citations.Add($"[{citations.Count + 1}] {cite.Title}: \"{cite.Snippet?.Trim()}\"");
                answer.Append($"[{citations.Count}]");
            }
        }
        return new RagResult(answer.ToString().Trim(), docs, new Dictionary<string, object?> { ["citations"] = citations });
    }
}
