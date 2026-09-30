using System.Text.Json.Serialization;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using RagCore;

namespace RagTypes;

/// <summary>
/// Adaptive RAG: a router agent classifies each question and picks the cheapest strategy that can
/// answer it: no retrieval, one lookup, or multi-step (split into sub-questions, answer each, combine).
/// </summary>
public sealed class AdaptiveRag(LlmProvider llm, IEmbeddingGenerator<string, Embedding<float>> embedder) : RagPipeline(llm, embedder)
{
    [JsonConverter(typeof(JsonStringEnumConverter<Strategy>))]
    public enum Strategy { SingleStep, MultiStep, NoRetrieval }

    public sealed record Route(Strategy Strategy, string Reason);
    public sealed record SubQuestions(List<string> Questions);

    private VectorStore _store = null!;
    private ChatClientAgent _router = null!;
    private ChatClientAgent _decomposer = null!;
    private ChatClientAgent _synthesizer = null!;
    private ChatClientAgent _generalAnswerer = null!;

    public override string Name => "adaptive";
    public override string Summary => "Query-complexity router over three strategies.";

    public override async Task IndexAsync(IReadOnlyList<Document> docs, CancellationToken ct = default)
    {
        _store = new VectorStore(Embedder);
        await _store.AddAsync(Chunker.Sentences(docs), ct: ct);
        _router = CreateAgent("router",
            "Choose a strategy for answering the question against a company knowledge base:\n" +
            "- NoRetrieval: general knowledge or chit-chat, no lookup needed\n" +
            "- SingleStep: one lookup answers it\n" +
            "- MultiStep: needs several facts combined, comparisons, or chained reasoning");
        _decomposer = CreateAgent("decomposer", "Break the question into 2-4 simple, self-contained sub-questions.");
        _synthesizer = CreateAgent("synthesizer", "Using the answered sub-questions, write a complete answer to the original question.");
        _generalAnswerer = CreateAgent("general-answerer");
    }

    private async Task<RagResult> SingleStepAsync(string question, Dictionary<string, object?> details, CancellationToken ct) =>
        await AnswerFromAsync(question, (await _store.SearchAsync(question, TopK, ct)).Select(h => h.Doc).ToList(), details, ct);

    public override async Task<RagResult> QueryAsync(string question, CancellationToken ct = default)
    {
        var route = await AskForAsync<Route>(_router, $"<question>\n{question}\n</question>", ct);
        var details = new Dictionary<string, object?> { ["strategy"] = route.Strategy, ["reason"] = route.Reason };

        switch (route.Strategy)
        {
            case Strategy.NoRetrieval:
                return new RagResult(await AskAsync(_generalAnswerer, question, ct: ct), [], details);

            case Strategy.MultiStep:
                var subs = (await AskForAsync<SubQuestions>(_decomposer, $"<question>\n{question}\n</question>", ct)).Questions ?? [];
                if (subs.Count == 0) subs = [question];
                var notes = new List<string>();
                var sources = new List<Document>();
                foreach (var sub in subs)
                {
                    var partial = await SingleStepAsync(sub, [], ct);
                    notes.Add($"Q: {sub}\nA: {partial.Answer}");
                    sources.AddRange(partial.Sources.Where(d => sources.All(s => s.Id != d.Id)));
                }
                var answer = await AskAsync(_synthesizer,
                    $"<context>\n{string.Join("\n\n", notes)}\n</context>\n\n<question>\n{question}\n</question>", ct: ct);
                details["sub_questions"] = subs;
                return new RagResult(answer, sources, details);

            default:
                return await SingleStepAsync(question, details, ct);
        }
    }
}
