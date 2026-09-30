using System.ComponentModel;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using RagCore;

namespace RagTypes;

/// <summary>
/// Self-RAG: decides whether retrieval is needed, answers, then a critic agent checks the answer:
/// is every claim supported by the context, and does it answer the question? If not, it searches
/// again with a better query and retries.
/// </summary>
public sealed class SelfRag(LlmProvider llm, IEmbeddingGenerator<string, Embedding<float>> embedder, int maxIterations = 3) : RagPipeline(llm, embedder)
{
    public sealed record RetrievalDecision(bool NeedsRetrieval);

    public sealed record Critique(
        [property: Description("Every claim is supported by the context")] bool Grounded,
        bool AnswersQuestion,
        List<string> UnsupportedClaims,
        [property: Description("A better search query to find what is missing")] string BetterSearchQuery);

    private VectorStore _store = null!;
    private ChatClientAgent _router = null!;
    private ChatClientAgent _critic = null!;
    private ChatClientAgent _generalAnswerer = null!;

    public override string Name => "self_rag";
    public override string Summary => "Retrieve-on-demand + generate + self-critique loop.";

    public override async Task IndexAsync(IReadOnlyList<Document> docs, CancellationToken ct = default)
    {
        _store = new VectorStore(Embedder);
        await _store.AddAsync(Chunker.Sentences(docs), ct: ct);
        _router = CreateAgent("retrieval-decider",
            "Decide whether answering the question requires looking up facts in the company knowledge base " +
            "(products, policies, people), or whether it is general knowledge or chit-chat.");
        _critic = CreateAgent("critic",
            "Critique the answer strictly against the context. If it is not grounded or incomplete, suggest a better search query.");
        _generalAnswerer = CreateAgent("general-answerer");
    }

    public override async Task<RagResult> QueryAsync(string question, CancellationToken ct = default)
    {
        var decision = await AskForAsync<RetrievalDecision>(_router, $"<question>\n{question}\n</question>", ct);
        if (!decision.NeedsRetrieval)
            return new RagResult(await AskAsync(_generalAnswerer, question, ct: ct), [], new Dictionary<string, object?> { ["retrieval"] = false });

        var searchQuery = question;
        var docs = new List<Document>();
        var trace = new List<string>();
        RagResult result = null!;
        for (var iteration = 1; iteration <= maxIterations; iteration++)
        {
            var seen = docs.Select(d => d.Id).ToHashSet();
            docs.AddRange((await _store.SearchAsync(searchQuery, TopK, ct)).Select(h => h.Doc).Where(d => !seen.Contains(d.Id)));
            result = await AnswerFromAsync(question, docs, ct: ct);

            var critique = await AskForAsync<Critique>(_critic,
                $"{AnswerPrompt(question, docs)}\n\n<answer>\n{result.Answer}\n</answer>", ct);
            trace.Add($"#{iteration} query='{searchQuery}' grounded={critique.Grounded} answers={critique.AnswersQuestion}");
            if (critique.Grounded && critique.AnswersQuestion)
                break;
            searchQuery = string.IsNullOrWhiteSpace(critique.BetterSearchQuery) ? question : critique.BetterSearchQuery;
        }
        return result with { Details = new Dictionary<string, object?> { ["retrieval"] = true, ["trace"] = trace } };
    }
}
