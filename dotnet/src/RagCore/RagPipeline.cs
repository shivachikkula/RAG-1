using System.Text;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;

namespace RagCore;

/// <summary>What a pipeline returns: the answer, the chunks it used, and pipeline-specific details.</summary>
public sealed record RagResult(string Answer, IReadOnlyList<Document> Sources, IReadOnlyDictionary<string, object?> Details)
{
    public RagResult(string answer, IReadOnlyList<Document> sources) : this(answer, sources, new Dictionary<string, object?>()) { }

    public string Pretty()
    {
        var sb = new StringBuilder(Answer).AppendLine().AppendLine().AppendLine("Sources:");
        for (var i = 0; i < Sources.Count; i++)
            sb.AppendLine($"  [{i + 1}] {Sources[i].Id} ({Sources[i].Source})");
        foreach (var (key, value) in Details)
            sb.AppendLine($"{key}: {Format(value)}");
        return sb.ToString().TrimEnd();
    }

    private static string Format(object? value) => value switch
    {
        string s => s,
        System.Collections.IEnumerable items => "[" + string.Join(", ", items.Cast<object?>().Select(Format)) + "]",
        _ => value?.ToString() ?? "",
    };
}

/// <summary>
/// Base class for every RAG type. Each type implements two steps:
/// <see cref="IndexAsync"/> (prepare the documents) and <see cref="QueryAsync"/> (answer a question).
/// Every LLM step is a Microsoft Agent Framework <see cref="ChatClientAgent"/>.
/// </summary>
public abstract class RagPipeline(LlmProvider llm, IEmbeddingGenerator<string, Embedding<float>> embedder, int topK = 4)
{
    public const string AnswerInstructions =
        "You answer questions using only the provided context. Cite sources inline as [n] using the " +
        "context numbers. If the context does not contain the answer, say you don't know.";

    private ChatClientAgent? _answerAgent;

    public abstract string Name { get; }

    /// <summary>One line describing how this RAG type works.</summary>
    public abstract string Summary { get; }

    public LlmProvider Llm { get; } = llm;
    protected IEmbeddingGenerator<string, Embedding<float>> Embedder { get; } = embedder;
    protected int TopK { get; } = topK;

    public abstract Task IndexAsync(IReadOnlyList<Document> docs, CancellationToken ct = default);

    public abstract Task<RagResult> QueryAsync(string question, CancellationToken ct = default);

    /// <summary>Creates an agent: a model plus instructions, and optionally tools it may call.</summary>
    protected ChatClientAgent CreateAgent(string name, string? instructions = null, IList<AITool>? tools = null, ChatOptions? options = null)
    {
        options ??= Llm.NewChatOptions();
        options.Instructions = instructions;
        options.Tools = tools;
        return Llm.ChatClient.AsAIAgent(new ChatClientAgentOptions { Name = name, ChatOptions = options });
    }

    /// <summary>Runs an agent and returns its text answer.</summary>
    protected static async Task<string> AskAsync(AIAgent agent, string prompt, AgentSession? session = null, CancellationToken ct = default) =>
        CheckRefusal(await agent.RunAsync(prompt, session, cancellationToken: ct)).Text.Trim();

    /// <summary>Runs an agent with structured output: the reply is parsed into <typeparamref name="T"/>.</summary>
    protected static async Task<T> AskForAsync<T>(AIAgent agent, string prompt, CancellationToken ct = default)
    {
        var response = await agent.RunAsync<T>(prompt, cancellationToken: ct);
        CheckRefusal(response);
        return response.Result;
    }

    protected static AgentResponse CheckRefusal(AgentResponse response) =>
        response.FinishReason == ChatFinishReason.ContentFilter
            ? throw new LlmRefusalException("Claude declined the request, and so did the fallback model.")
            : response;

    public static string AnswerPrompt(string question, IEnumerable<Document> docs) =>
        $"<context>\n{Chunker.FormatContext(docs)}\n</context>\n\n<question>\n{question}\n</question>";

    /// <summary>The common last step: answer the question from these chunks, with [n] citations.</summary>
    protected async Task<RagResult> AnswerFromAsync(string question, IReadOnlyList<Document> docs, IReadOnlyDictionary<string, object?>? details = null, CancellationToken ct = default)
    {
        _answerAgent ??= CreateAgent("answerer", AnswerInstructions);
        var answer = await AskAsync(_answerAgent, AnswerPrompt(question, docs), ct: ct);
        return new RagResult(answer, docs, details ?? new Dictionary<string, object?>());
    }
}
