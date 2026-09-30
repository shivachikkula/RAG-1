using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using RagCore;

namespace RagTypes;

/// <summary>
/// Conversational RAG: remembers the chat. A follow-up like "what about its battery?" is first
/// rewritten into a standalone question for search. The answering agent keeps the conversation in
/// an Agent Framework <see cref="AgentSession"/>, so it sees earlier turns automatically.
/// </summary>
public sealed class ConversationalRag(LlmProvider llm, IEmbeddingGenerator<string, Embedding<float>> embedder, int maxTurns = 6) : RagPipeline(llm, embedder)
{
    public sealed record Standalone(string StandaloneQuestion);

    private readonly List<(string Question, string Answer)> _turns = [];
    private VectorStore _store = null!;
    private ChatClientAgent _condenser = null!;
    private ChatClientAgent _answerer = null!;
    private AgentSession? _session;

    public override string Name => "conversational";
    public override string Summary => "History-aware query condensation + agent session memory.";

    public override async Task IndexAsync(IReadOnlyList<Document> docs, CancellationToken ct = default)
    {
        _store = new VectorStore(Embedder);
        await _store.AddAsync(Chunker.Sentences(docs), ct: ct);
        _condenser = CreateAgent("condenser",
            "Rewrite the latest user question as a standalone question that can be understood without the conversation, resolving pronouns and references.");
        _answerer = CreateAgent("chat-answerer", AnswerInstructions);
    }

    private async Task<string> CondenseAsync(string question, CancellationToken ct)
    {
        if (_turns.Count == 0)
            return question;
        var history = string.Join("\n", _turns.TakeLast(maxTurns).Select(t => $"User: {t.Question}\nAssistant: {t.Answer}"));
        var result = await AskForAsync<Standalone>(_condenser,
            $"<conversation>\n{history}\n</conversation>\n\n<question>\n{question}\n</question>", ct);
        return string.IsNullOrWhiteSpace(result.StandaloneQuestion) ? question : result.StandaloneQuestion;
    }

    public override async Task<RagResult> QueryAsync(string question, CancellationToken ct = default)
    {
        var standalone = await CondenseAsync(question, ct);
        var docs = (await _store.SearchAsync(standalone, TopK, ct)).Select(h => h.Doc).ToList();

        _session ??= await _answerer.CreateSessionAsync(ct);  // holds the chat history between calls
        var answer = await AskAsync(_answerer, AnswerPrompt(question, docs), _session, ct);
        _turns.Add((question, answer));
        return new RagResult(answer, docs, new Dictionary<string, object?> { ["standalone_question"] = standalone, ["turn"] = _turns.Count });
    }

    public void Reset()
    {
        _turns.Clear();
        _session = null;
    }
}
