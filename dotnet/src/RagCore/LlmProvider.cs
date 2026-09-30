using Anthropic;
using Anthropic.Models.Beta.Messages;
using Microsoft.Extensions.AI;

namespace RagCore;

public enum LlmMode
{
    /// <summary>Claude when ANTHROPIC_API_KEY is set, otherwise the offline mock.</summary>
    Auto,
    /// <summary>Always the offline mock. No API key, no network.</summary>
    Mock,
    /// <summary>Always Claude. Fails if no API key is set.</summary>
    Live,
}

/// <summary>
/// The model every pipeline talks to, as a standard .NET <see cref="IChatClient"/>.
/// Microsoft Agent Framework builds agents on top of any IChatClient, so the RAG code
/// never depends on which model is behind it.
/// </summary>
public sealed class LlmProvider
{
    public const string DefaultModel = "claude-opus-5";

    // Server-side refusal fallback: if Claude declines a request, the API re-runs it on
    // Anthropic's recommended fallback model inside the same call.
    private const string FallbackBeta = "server-side-fallback-2026-07-01";

    private LlmProvider(IChatClient chatClient, string modelName, bool isMock)
    {
        ChatClient = chatClient;
        ModelName = modelName;
        IsMock = isMock;
    }

    public IChatClient ChatClient { get; }
    public string ModelName { get; }
    public bool IsMock { get; }

    public static LlmProvider Create(LlmMode mode = LlmMode.Auto)
    {
        var hasKey = !string.IsNullOrEmpty(Environment.GetEnvironmentVariable("ANTHROPIC_API_KEY"));
        if (mode == LlmMode.Live && !hasKey)
            throw new InvalidOperationException("Live mode needs the ANTHROPIC_API_KEY environment variable.");
        if (mode == LlmMode.Mock || (mode == LlmMode.Auto && !hasKey))
            return CreateMock();
        return CreateClaude(new AnthropicClient());
    }

    public static LlmProvider CreateMock() => new(new MockChatClient(), "mock", isMock: true);

    /// <summary>Wraps an Anthropic client (the beta surface, for refusal fallbacks) as an IChatClient.</summary>
    public static LlmProvider CreateClaude(AnthropicClient client, string? model = null)
    {
        model ??= Environment.GetEnvironmentVariable("RAG_MODEL") ?? DefaultModel;
        return new(client.Beta.AsIChatClient(model, 16000), model, isMock: false);
    }

    /// <summary>
    /// Fresh chat options for one agent. For Claude, the request "template" turns on refusal
    /// fallbacks and can add Claude-specific extras the generic abstraction has no field for:
    /// server tools (web search), cached system blocks, or document blocks with citations.
    /// </summary>
    public ChatOptions NewChatOptions(
        IReadOnlyList<BetaToolUnion>? serverTools = null,
        IReadOnlyList<BetaTextBlockParam>? system = null,
        IReadOnlyList<BetaMessageParam>? leadingMessages = null)
    {
        if (IsMock)
            return new ChatOptions();

        return new ChatOptions
        {
            RawRepresentationFactory = _ =>
            {
                var request = new MessageCreateParams
                {
                    Model = ModelName,
                    MaxTokens = 16000,
                    Messages = leadingMessages ?? [],
                    Betas = [FallbackBeta],
                    Fallbacks = new Default(),
                };
                return request with
                {
                    Tools = serverTools ?? request.Tools,
                    System = system is null ? request.System : system.ToList(),
                };
            },
        };
    }
}

public sealed class LlmRefusalException(string message) : Exception(message);
