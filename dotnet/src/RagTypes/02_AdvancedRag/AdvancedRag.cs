using System.ComponentModel;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using RagCore;

namespace RagTypes;

/// <summary>
/// Advanced RAG: improves the steps around retrieval. Before: an agent rewrites the question into a
/// better search query. After: more candidates are retrieved, and an agent reranks them.
/// </summary>
public sealed class AdvancedRag(LlmProvider llm, IEmbeddingGenerator<string, Embedding<float>> embedder) : RagPipeline(llm, embedder)
{
    public sealed record SearchQuery([property: Description("Concise, keyword-rich search query")] string Query);
    public sealed record PassageScore(int Passage, [property: Description("0 = irrelevant, 10 = directly answers the question")] int Relevance);
    public sealed record Ranking(List<PassageScore> Scores);

    private VectorStore _store = null!;
    private ChatClientAgent _rewriter = null!;
    private ChatClientAgent _reranker = null!;

    public override string Name => "advanced";
    public override string Summary => "Query rewrite -> over-retrieve -> LLM rerank -> answer.";

    public override async Task IndexAsync(IReadOnlyList<Document> docs, CancellationToken ct = default)
    {
        _store = new VectorStore(Embedder);
        await _store.AddAsync(Chunker.Sentences(docs, maxChars: 400, overlapSentences: 1), ct: ct);
        _rewriter = CreateAgent("rewriter", "Rewrite the user's question into a concise, keyword-rich search query. Expand abbreviations and drop filler words.");
        _reranker = CreateAgent("reranker", "Score how useful each passage is for answering the question. Return one entry per passage.");
    }

    public override async Task<RagResult> QueryAsync(string question, CancellationToken ct = default)
    {
        var rewritten = (await AskForAsync<SearchQuery>(_rewriter, $"<question>\n{question}\n</question>", ct)).Query;
        var searchQuery = string.IsNullOrWhiteSpace(rewritten) ? question : rewritten;

        var candidates = (await _store.SearchAsync(searchQuery, k: 10, ct)).Select(h => h.Doc).ToList();
        var passages = string.Join("\n\n", candidates.Select((d, i) => $"<passage id=\"{i}\">\n{d.Text}\n</passage>"));
        var ranking = await AskForAsync<Ranking>(_reranker, $"{passages}\n\n<question>\n{question}\n</question>", ct);

        var scores = (ranking.Scores ?? []).Where(s => s.Passage >= 0 && s.Passage < candidates.Count).ToDictionary(s => s.Passage, s => s.Relevance);
        var top = scores.Count == 0  // e.g. the mock LLM: keep the retrieval order
            ? candidates
            : candidates.Select((d, i) => (d, i)).Where(x => scores.GetValueOrDefault(x.i) > 0).OrderByDescending(x => scores[x.i]).Select(x => x.d).ToList();

        return await AnswerFromAsync(question, top.Take(TopK).ToList(), new Dictionary<string, object?> { ["search_query"] = searchQuery }, ct);
    }
}
