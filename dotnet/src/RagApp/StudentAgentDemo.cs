using System.Text.Encodings.Web;
using System.Text.Json;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using ModelContextProtocol.Client;
using ModelContextProtocol.Protocol;
using RagCore;

/// <summary>
/// The 'students' command: starts the Student MCP server as a separate process, connects to it as
/// an MCP client, and gives its tools to an Agent Framework agent. The agent decides which tools
/// to call (database, file storage, external API) to answer the question.
/// </summary>
internal static class StudentAgentDemo
{
    public const string DefaultQuestion = "How is Aisha Khan doing, what do her advisor notes say, and what should she focus on next?";

    public static async Task<int> RunAsync(LlmProvider llm, string question)
    {
        var project = FindServerProject();
        Console.WriteLine($"Starting the Student MCP server ({Path.GetFileName(project)})...");
        await using var mcp = await McpClient.CreateAsync(new StdioClientTransport(new StdioClientTransportOptions
        {
            Name = "student-mcp",
            Command = "dotnet",
            Arguments = ["run", "--project", project],
            StandardErrorLines = _ => { },  // the server's own logs; hide them
        }));

        var tools = await mcp.ListToolsAsync();
        Console.WriteLine($"Connected. {tools.Count} tools: {string.Join(", ", tools.Select(t => t.Name))}\n");

        if (llm.IsMock)
        {
            // The offline mock can't decide which tools to call, so call one directly to show the data flowing.
            Console.WriteLine("Mock mode: calling get_student_overview for S1001 directly. Use --live to let Claude pick the tools.\n");
            var result = await mcp.CallToolAsync("get_student_overview", new Dictionary<string, object?> { ["studentId"] = "S1001" });
            var text = string.Concat(result.Content.OfType<TextContentBlock>().Select(b => b.Text));
            Console.WriteLine(JsonSerializer.Serialize(JsonDocument.Parse(text), new JsonSerializerOptions { WriteIndented = true, Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping }));
            return 0;
        }

        // MCP tools are AIFunctions, so they plug straight into an agent like any other tool.
        var options = llm.NewChatOptions();
        options.Instructions = mcp.ServerInstructions;
        options.Tools = [.. tools];
        var agent = llm.ChatClient.AsAIAgent(new ChatClientAgentOptions { Name = "student-advisor", ChatOptions = options });

        Console.WriteLine($"Q: {question}\n");
        var response = await agent.RunAsync(question);
        foreach (var call in response.Messages.SelectMany(m => m.Contents).OfType<FunctionCallContent>())
            Console.WriteLine($"  tool call: {call.Name}({JsonSerializer.Serialize(call.Arguments)})");
        Console.WriteLine($"\n{response.Text}");
        return 0;
    }

    private static string FindServerProject()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            var candidate = Path.Combine(dir.FullName, "src", "StudentMcpServer", "StudentMcpServer.csproj");
            if (File.Exists(candidate)) return candidate;
        }
        throw new FileNotFoundException("Could not find src/StudentMcpServer/StudentMcpServer.csproj.");
    }
}
