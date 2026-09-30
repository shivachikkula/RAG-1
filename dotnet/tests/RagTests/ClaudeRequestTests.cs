using System.Net;
using System.Text;
using System.Text.Json.Nodes;
using Anthropic;
using RagCore;
using RagTypes;

namespace RagTests;

/// <summary>
/// Checks the requests the Claude path actually sends, using a fake HTTP handler instead of the
/// network, so no API key is needed.
/// </summary>
public class ClaudeRequestTests
{
    private static readonly List<Document> Docs = DocumentLoader.LoadDirectory(DocumentLoader.FindSampleData());

    /// <summary>Records every request and answers with a canned Claude response.</summary>
    private sealed class FakeClaude(Func<JsonObject, int, JsonArray> reply) : HttpMessageHandler
    {
        public List<JsonObject> Requests { get; } = [];
        public List<string> BetaHeaders { get; } = [];

        protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken ct)
        {
            var body = JsonNode.Parse(await request.Content!.ReadAsStringAsync(ct))!.AsObject();
            Requests.Add(body);
            BetaHeaders.Add(request.Headers.TryGetValues("anthropic-beta", out var v) ? string.Join(",", v) : "");
            var content = reply(body, Requests.Count);
            var stop = content.Any(c => c!["type"]!.GetValue<string>() == "tool_use") ? "tool_use" : "end_turn";
            var json = new JsonObject
            {
                ["id"] = $"msg_{Requests.Count}", ["type"] = "message", ["role"] = "assistant", ["model"] = "claude-opus-5",
                ["content"] = content, ["stop_reason"] = stop, ["stop_sequence"] = null,
                ["usage"] = new JsonObject { ["input_tokens"] = 1, ["output_tokens"] = 1 },
            };
            return new HttpResponseMessage(HttpStatusCode.OK) { Content = new StringContent(json.ToJsonString(), Encoding.UTF8, "application/json") };
        }
    }

    private static JsonArray Text(string text) => [new JsonObject { ["type"] = "text", ["text"] = text }];

    private static (LlmProvider, FakeClaude) Claude(Func<JsonObject, int, JsonArray>? reply = null)
    {
        var fake = new FakeClaude(reply ?? ((_, _) => Text("10 hours [1].")));
        var client = new AnthropicClient(new() { ApiKey = "test-key", HttpClient = new HttpClient(fake) });
        return (LlmProvider.CreateClaude(client, "claude-opus-5"), fake);
    }

    [Fact]
    public async Task RequestsUseOpusWithRefusalFallback()
    {
        var (llm, fake) = Claude();
        var rag = new NaiveRag(llm, new HashingEmbeddingGenerator());
        await rag.IndexAsync(Docs);
        var result = await rag.QueryAsync("How long does the Stratus S1 battery last?");

        Assert.Equal("10 hours [1].", result.Answer);
        var request = Assert.Single(fake.Requests);
        Assert.Equal("claude-opus-5", request["model"]!.GetValue<string>());
        Assert.Equal("default", request["fallbacks"]!.GetValue<string>());
        Assert.Contains("server-side-fallback-2026-07-01", fake.BetaHeaders[0]);
        Assert.Contains("only the provided context", request["system"]!.ToJsonString());
        Assert.False(request.ContainsKey("tools"));
    }

    [Fact]
    public async Task StructuredOutputSendsJsonSchema()
    {
        var (llm, fake) = Claude((body, _) =>
        {
            var schema = body["output_config"]?["format"]?["schema"]?.ToJsonString() ?? "";
            if (schema.Contains("\"query\"")) return Text("""{"query":"Stratus S1 battery life hours"}""");  // rewriter
            if (schema.Contains("\"scores\"")) return Text("""{"scores":[{"passage":0,"relevance":9}]}""");   // reranker
            return Text("10 hours.");
        });
        var rag = new AdvancedRag(llm, new HashingEmbeddingGenerator());
        await rag.IndexAsync(Docs);
        var result = await rag.QueryAsync("how long does the big robot run?");

        Assert.Equal("json_schema", fake.Requests[0]["output_config"]!["format"]!["type"]!.GetValue<string>());
        Assert.Equal("Stratus S1 battery life hours", result.Details["search_query"]);
    }

    [Fact]
    public async Task AgenticExecutesClaudeToolCalls()
    {
        var (llm, fake) = Claude((_, call) => call == 1
            ? [new JsonObject { ["type"] = "tool_use", ["id"] = "toolu_1", ["name"] = "search_knowledge_base", ["input"] = new JsonObject { ["query"] = "battery", ["k"] = 3 } }]
            : Text("It runs for 10 hours [products#0]."));
        var rag = new AgenticRag(llm, new HashingEmbeddingGenerator());
        await rag.IndexAsync(Docs);
        var result = await rag.QueryAsync("How long does the Stratus S1 battery last?");

        Assert.Equal(2, fake.Requests.Count);
        var tools = fake.Requests[0]["tools"]!.AsArray().Select(t => t!["name"]!.GetValue<string>()).ToList();
        Assert.Equal(["search_knowledge_base", "list_documents", "read_document"], tools);
        Assert.Contains("tool_result", fake.Requests[1]["messages"]!.ToJsonString());
        Assert.NotEmpty(result.Sources);
    }

    [Fact]
    public async Task CitationSendsDocumentsAndReadsCitations()
    {
        var (llm, fake) = Claude((_, _) =>
        [
            new JsonObject
            {
                ["type"] = "text", ["text"] = "10 hours",
                ["citations"] = new JsonArray(new JsonObject
                {
                    ["type"] = "char_location", ["cited_text"] = "runs for 10 hours", ["document_index"] = 0,
                    ["document_title"] = "products#0", ["start_char_index"] = 0, ["end_char_index"] = 17,
                }),
            },
        ]);
        var rag = new CitationRag(llm, new HashingEmbeddingGenerator());
        await rag.IndexAsync(Docs);
        var result = await rag.QueryAsync("How long does the Stratus S1 battery last?");

        var firstBlock = fake.Requests[0]["messages"]![0]!["content"]![0]!;
        Assert.Equal("document", firstBlock["type"]!.GetValue<string>());
        Assert.True(firstBlock["citations"]!["enabled"]!.GetValue<bool>());
        Assert.Equal("10 hours[1]", result.Answer);
        Assert.Contains("runs for 10 hours", Assert.IsType<List<string>>(result.Details["citations"])[0]);
    }

    [Fact]
    public async Task CorrectiveFallsBackToWebSearch()
    {
        var (llm, fake) = Claude((body, _) => body.ContainsKey("output_config")
            ? Text("""{"items":[],"sufficient":false}""")
            : Text("From the web: ..."));
        var rag = new CorrectiveRag(llm, new HashingEmbeddingGenerator());
        await rag.IndexAsync(Docs);
        var result = await rag.QueryAsync("What does ISO 3691-4 cover?");

        Assert.Equal("web_search", result.Details["action"]);
        Assert.Equal("web_search_20260209", fake.Requests[^1]["tools"]![0]!["type"]!.GetValue<string>());
    }

    [Fact]
    public async Task ContextualRetrievalCachesTheDocument()
    {
        var (llm, fake) = Claude((_, _) => Text("Context."));
        var rag = new ContextualRetrievalRag(llm, new HashingEmbeddingGenerator());
        await rag.IndexAsync(Docs[..1]);

        var system = fake.Requests[0]["system"]!.AsArray();
        Assert.Contains(system, block => block!["cache_control"]?["type"]?.GetValue<string>() == "ephemeral"
                                         && block["text"]!.GetValue<string>().StartsWith("<document>"));
    }
}
