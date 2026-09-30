using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using Microsoft.Extensions.AI;

namespace RagCore;

/// <summary>
/// An offline stand-in for Claude so every pipeline and test runs with no API key.
/// It is not intelligent:
/// <list type="bullet">
/// <item>Plain questions: returns the context sentences that share the most words with the question.</item>
/// <item>Structured output (a JSON schema): returns a schema-valid default value.</item>
/// <item>Tools: if a <c>search_knowledge_base</c> tool is offered, it calls it once, so the
/// Agent Framework tool loop really runs offline.</item>
/// </list>
/// </summary>
public sealed partial class MockChatClient : IChatClient
{
    public Task<ChatResponse> GetResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null, CancellationToken cancellationToken = default)
    {
        var history = messages.ToList();
        var question = ExtractQuestion(history);

        if (options?.ResponseFormat is ChatResponseFormatJson { Schema: { } schema })
            return Reply(Fill(schema, question)!.ToJsonString());

        var toolResults = history.SelectMany(m => m.Contents).OfType<FunctionResultContent>().ToList();
        if (toolResults.Count == 0 && options?.Tools?.OfType<AIFunction>().Any(t => t.Name == "search_knowledge_base") == true)
        {
            var call = new FunctionCallContent("mock_call_1", "search_knowledge_base",
                new Dictionary<string, object?> { ["query"] = question, ["k"] = 4 });
            return Task.FromResult(new ChatResponse(new ChatMessage(ChatRole.Assistant, [call])));
        }

        var context = string.Join("\n", history.Select(m => Section(m.Text, "context")).Where(s => s.Length > 0)
            .Concat(toolResults.Select(r => ToolResultText(r.Result))));
        return Reply(context.Length == 0 ? $"[mock] {question}" : "[mock] " + BestSentences(context, question));
    }

    public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        foreach (var update in (await GetResponseAsync(messages, options, cancellationToken)).ToChatResponseUpdates())
            yield return update;
    }

    public object? GetService(Type serviceType, object? serviceKey = null) => serviceType.IsInstanceOfType(this) ? this : null;

    public void Dispose() { }

    private static Task<ChatResponse> Reply(string text) =>
        Task.FromResult(new ChatResponse(new ChatMessage(ChatRole.Assistant, text)) { FinishReason = ChatFinishReason.Stop });

    private static string ExtractQuestion(List<ChatMessage> history)
    {
        var lastUser = history.LastOrDefault(m => m.Role == ChatRole.User)?.Text ?? "";
        var tagged = Section(lastUser, "question");
        return tagged.Length > 0 ? tagged : lastUser.Trim().Split('\n')[^1];
    }

    private static string Section(string text, string tag)
    {
        var match = Regex.Match(text, $@"<{tag}>\s*(.*?)\s*</{tag}>", RegexOptions.Singleline);
        return match.Success ? match.Groups[1].Value : "";
    }

    /// <summary>Tool results are JSON arrays of chunks; pull out their "text" fields.</summary>
    private static string ToolResultText(object? result)
    {
        var raw = result is JsonElement element && element.ValueKind == JsonValueKind.String ? element.GetString() : result?.ToString();
        try
        {
            var chunks = JsonNode.Parse(raw ?? "") as JsonArray;
            if (chunks is not null)
                return string.Join("\n", chunks.Select(c => c?["text"]?.GetValue<string>()).Where(t => t is not null));
        }
        catch (JsonException) { }
        return raw ?? "";
    }

    private static string BestSentences(string context, string question)
    {
        var queryTerms = TextUtil.Tokenize(question).ToHashSet();
        var sentences = SentenceSplit().Split(context)
            .Select(s => s.Trim().Trim('"', ',', '{', '}', '[', ']'))
            .Where(s => s.Length > 20 && !s.StartsWith('['))
            .Distinct();
        return string.Join(' ', sentences.OrderByDescending(s => TextUtil.Tokenize(s).Count(queryTerms.Contains)).Take(2));
    }

    [GeneratedRegex(@"(?<=[.!?])\s+|\n+|\\n")]
    private static partial Regex SentenceSplit();

    /// <summary>Builds a minimal value that satisfies a JSON schema.</summary>
    private static JsonNode? Fill(JsonElement schema, string question)
    {
        if (schema.TryGetProperty("enum", out var values))
            return JsonNode.Parse(values[0].GetRawText());
        var type = schema.TryGetProperty("type", out var t)
            ? t.ValueKind == JsonValueKind.Array ? t.EnumerateArray().First(x => x.GetString() != "null").GetString() : t.GetString()
            : "object";
        switch (type)
        {
            case "object":
                var obj = new JsonObject();
                if (schema.TryGetProperty("properties", out var props))
                    foreach (var p in props.EnumerateObject())
                        obj[p.Name] = Fill(p.Value, question);
                return obj;
            case "array":
                var isStringArray = schema.TryGetProperty("items", out var items) && items.TryGetProperty("type", out var it) && it.ValueKind == JsonValueKind.String && it.GetString() == "string";
                return isStringArray ? new JsonArray(question) : new JsonArray();
            case "boolean":
                return true;
            case "integer":
            case "number":
                return 1;
            default:
                return question;
        }
    }
}
