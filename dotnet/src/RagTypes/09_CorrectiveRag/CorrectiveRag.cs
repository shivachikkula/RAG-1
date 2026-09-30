using System.ComponentModel;
using Anthropic.Models.Beta.Messages;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using RagCore;

namespace RagTypes;

/// <summary>
/// Corrective RAG (CRAG): a grader agent checks every retrieved chunk for relevance and drops the
/// irrelevant ones. If the knowledge base isn't enough, it falls back to Claude's server-side web
/// search instead of answering from noise.
/// </summary>
public sealed class CorrectiveRag(LlmProvider llm, IEmbeddingGenerator<string, Embedding<float>> embedder) : RagPipeline(llm, embedder)
{
    public sealed record PassageGrade(int Passage, bool Relevant);
    public sealed record Grades(List<PassageGrade> Items, [property: Description("Do the relevant passages fully answer the question?")] bool Sufficient);

    private VectorStore _store = null!;
    private ChatClientAgent _grader = null!;

    public override string Name => "corrective";
    public override string Summary => "Retrieve -> grade -> (filter | web-search fallback) -> answer.";

    public override async Task IndexAsync(IReadOnlyList<Document> docs, CancellationToken ct = default)
    {
        _store = new VectorStore(Embedder);
        await _store.AddAsync(Chunker.Sentences(docs), ct: ct);
        _grader = CreateAgent("grader",
            "For each passage decide whether it contains information relevant to the question. " +
            "Then decide whether the relevant passages together are sufficient to answer it.");
    }

    private async Task<string> WebAnswerAsync(string question, IReadOnlyList<Document> relevant, CancellationToken ct)
    {
        var options = Llm.NewChatOptions(serverTools: [new BetaWebSearchTool20260209 { MaxUses = 3 }]);
        var agent = CreateAgent("web-researcher", AnswerInstructions, options: options);
        var local = relevant.Count > 0 ? string.Join("\n\n", relevant.Select(d => d.Text)) : "(none)";
        return await AskAsync(agent,
            $"The internal knowledge base only had this:\n<context>\n{local}\n</context>\n\n" +
            $"Search the web to fill the gaps, then answer.\n<question>\n{question}\n</question>", ct: ct);
    }

    public override async Task<RagResult> QueryAsync(string question, CancellationToken ct = default)
    {
        var candidates = (await _store.SearchAsync(question, TopK * 2, ct)).Select(h => h.Doc).ToList();
        var passages = string.Join("\n\n", candidates.Select((d, i) => $"<passage id=\"{i}\">\n{d.Text}\n</passage>"));
        var grades = await AskForAsync<Grades>(_grader, $"{passages}\n\n<question>\n{question}\n</question>", ct);

        var irrelevant = (grades.Items ?? []).Where(g => !g.Relevant).Select(g => g.Passage).ToHashSet();
        var relevant = candidates.Where((_, i) => !irrelevant.Contains(i)).ToList();
        var details = new Dictionary<string, object?> { ["retrieved"] = candidates.Count, ["kept"] = relevant.Count, ["sufficient"] = grades.Sufficient };

        if (!grades.Sufficient && !Llm.IsMock)
        {
            details["action"] = "web_search";
            return new RagResult(await WebAnswerAsync(question, relevant, ct), relevant, details);
        }
        if (relevant.Count == 0)
            return new RagResult("I don't know - nothing relevant was found in the knowledge base.", [], details);
        details["action"] = "answer_from_kb";
        return await AnswerFromAsync(question, relevant.Take(TopK).ToList(), details, ct);
    }
}
