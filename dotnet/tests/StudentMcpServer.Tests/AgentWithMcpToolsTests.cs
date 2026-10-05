using System.Net;
using System.Text;
using System.Text.Json.Nodes;
using Anthropic;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using RagCore;

namespace StudentMcpServer.Tests;

/// <summary>
/// End to end: an Agent Framework agent gets the MCP server's tools, "Claude" (a fake HTTP handler)
/// asks to call one, the framework runs it through MCP, and the result goes back to Claude.
/// </summary>
public class AgentWithMcpToolsTests
{
    private sealed class FakeClaude : HttpMessageHandler
    {
        public List<JsonObject> Requests { get; } = [];

        protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken ct)
        {
            Requests.Add(JsonNode.Parse(await request.Content!.ReadAsStringAsync(ct))!.AsObject());
            JsonArray content = Requests.Count == 1
                ? [new JsonObject { ["type"] = "tool_use", ["id"] = "toolu_1", ["name"] = "get_student_transcript", ["input"] = new JsonObject { ["studentId"] = "S1002" } }]
                : [new JsonObject { ["type"] = "text", ["text"] = "Diego has a 3.11 GPA." }];
            var body = new JsonObject
            {
                ["id"] = $"msg_{Requests.Count}", ["type"] = "message", ["role"] = "assistant", ["model"] = "claude-opus-5",
                ["content"] = content, ["stop_reason"] = Requests.Count == 1 ? "tool_use" : "end_turn", ["stop_sequence"] = null,
                ["usage"] = new JsonObject { ["input_tokens"] = 1, ["output_tokens"] = 1 },
            };
            return new HttpResponseMessage(HttpStatusCode.OK) { Content = new StringContent(body.ToJsonString(), Encoding.UTF8, "application/json") };
        }
    }

    [Fact]
    public async Task AgentCallsMcpToolsThroughClaude()
    {
        await using var server = await TestServer.StartAsync();
        var mcpTools = await server.Client.ListToolsAsync();

        var fake = new FakeClaude();
        var llm = LlmProvider.CreateClaude(new AnthropicClient(new() { ApiKey = "test", HttpClient = new HttpClient(fake) }), "claude-opus-5");
        var options = llm.NewChatOptions();
        options.Instructions = server.Client.ServerInstructions;
        options.Tools = [.. mcpTools];
        var agent = llm.ChatClient.AsAIAgent(new ChatClientAgentOptions { Name = "student-advisor", ChatOptions = options });

        var response = await agent.RunAsync("What is Diego Martinez's GPA?");

        Assert.Equal("Diego has a 3.11 GPA.", response.Text);
        var sentTools = fake.Requests[0]["tools"]!.AsArray().Select(t => t!["name"]!.GetValue<string>()).ToList();
        Assert.Contains("get_student_transcript", sentTools);
        Assert.Equal(10, sentTools.Count);
        Assert.Contains("Student information service", fake.Requests[0]["system"]!.ToJsonString());
        // The second request carries the real tool result from the MCP server (database data).
        var toolResult = fake.Requests[1]["messages"]!.ToJsonString();
        Assert.Contains("tool_result", toolResult);
        Assert.Contains("Diego Martinez", toolResult);
        Assert.Contains("3.11", toolResult);
    }
}
