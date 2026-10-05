using System.Diagnostics;
using System.IO.Pipelines;
using System.Text.Encodings.Web;
using System.Text.Json;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using ModelContextProtocol.Client;
using ModelContextProtocol.Protocol;

namespace StudentMcpServer;

/// <summary>
/// An interactive console for trying and debugging the server: <c>--console</c>.
/// The server runs inside this same process, and a real MCP client talks to it through in-memory
/// pipes. So everything goes through the real protocol, and breakpoints in tool code are hit
/// when you start this under a debugger (F5).
/// </summary>
public static class McpConsole
{
    private static readonly JsonSerializerOptions Pretty = new() { WriteIndented = true, Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping };

    private const string Help = """
        Commands:
          tools                          list the tools and their parameters
          call <tool> [name=value ...]   call a tool, e.g.  call get_student studentId=S1001
          call <tool> {json}             same, with JSON:   call find_textbooks {"topic":"calculus","limit":3}
          resources                      list resource templates
          read <uri>                     read a resource, e.g.  read student://S1001/transcript
          prompts                        list prompts
          prompt <name> [name=value ...] show a prompt, e.g.  prompt study_plan studentId=S1003 courseCode=MATH150
          help                           show this help
          exit                           quit
        """;

    public static async Task<int> RunAsync(StudentMcpOptions options, LogLevel logLevel)
    {
        var clientToServer = new Pipe();
        var serverToClient = new Pipe();

        var builder = Host.CreateApplicationBuilder();
        builder.Logging.ClearProviders();
        builder.Logging.AddConsole(o => o.LogToStandardErrorThreshold = LogLevel.Trace);
        builder.Logging.SetMinimumLevel(logLevel);
        builder.Services
            .AddStudentMcpServer(options)
            .WithStreamServerTransport(clientToServer.Reader.AsStream(), serverToClient.Writer.AsStream());
        using var host = builder.Build();
        await host.StartAsync();

        await using var client = await McpClient.CreateAsync(
            new StreamClientTransport(clientToServer.Writer.AsStream(), serverToClient.Reader.AsStream()));

        Console.WriteLine($"Connected to {client.ServerInfo.Name} {client.ServerInfo.Version} (in-process).");
        Console.WriteLine($"Database: {options.DatabasePath}\nFiles:    {options.FilesPath}\n");
        Console.WriteLine(Help);

        while (true)
        {
            if (!Console.IsInputRedirected) Console.Write("\nmcp> ");
            var line = Console.ReadLine();
            if (line is null) break;  // end of piped input
            line = line.Trim();
            if (line.Length == 0 || line.StartsWith('#')) continue;
            if (Console.IsInputRedirected) Console.WriteLine($"\nmcp> {line}");

            var (command, rest) = Split(line);
            try
            {
                switch (command)
                {
                    case "exit" or "quit":
                        await host.StopAsync();
                        return 0;
                    case "help":
                        Console.WriteLine(Help);
                        break;
                    case "tools":
                        foreach (var tool in await client.ListToolsAsync())
                            Console.WriteLine($"{tool.Name}({Parameters(tool.JsonSchema)}){Flags(tool.ProtocolTool)}\n    {tool.Description}");
                        break;
                    case "call":
                        var (toolName, argText) = Split(rest);
                        var timer = Stopwatch.StartNew();
                        var result = await client.CallToolAsync(toolName, ParseArguments(argText));
                        PrintResult(result, timer.ElapsedMilliseconds);
                        break;
                    case "resources":
                        foreach (var template in await client.ListResourceTemplatesAsync())
                            Console.WriteLine($"{template.UriTemplate}\n    {template.Description}");
                        break;
                    case "read":
                        foreach (var content in (await client.ReadResourceAsync(rest)).Contents)
                            Console.WriteLine(content is TextResourceContents text ? PrettyIfJson(text.Text) : content.ToString());
                        break;
                    case "prompts":
                        foreach (var prompt in await client.ListPromptsAsync())
                            Console.WriteLine($"{prompt.Name}({string.Join(", ", prompt.ProtocolPrompt.Arguments?.Select(a => a.Name) ?? [])})\n    {prompt.Description}");
                        break;
                    case "prompt":
                        var (promptName, promptArgs) = Split(rest);
                        var got = await client.GetPromptAsync(promptName, ParseArguments(promptArgs));
                        foreach (var message in got.Messages)
                            Console.WriteLine($"[{message.Role}] {(message.Content as TextContentBlock)?.Text}");
                        break;
                    default:
                        Console.WriteLine($"Unknown command '{command}'. Type 'help'.");
                        break;
                }
            }
            catch (Exception ex) when (ex is not OperationCanceledException)
            {
                WriteColored(ConsoleColor.Red, $"{ex.GetType().Name}: {ex.Message}");
            }
        }

        await host.StopAsync();
        return 0;
    }

    private static void PrintResult(CallToolResult result, long ms)
    {
        var text = string.Concat(result.Content.OfType<TextContentBlock>().Select(b => b.Text));
        if (result.IsError == true)
            WriteColored(ConsoleColor.Red, $"Tool returned an error ({ms} ms):\n{text}");
        else
        {
            WriteColored(ConsoleColor.Green, $"OK ({ms} ms)");
            Console.WriteLine(PrettyIfJson(text));
        }
    }

    /// <summary>Accepts {json} or name=value pairs; numbers and true/false become numbers and booleans.</summary>
    private static Dictionary<string, object?> ParseArguments(string text)
    {
        text = text.Trim();
        if (text.Length == 0) return [];
        if (text.StartsWith('{'))
            return JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(text)!.ToDictionary(kv => kv.Key, kv => (object?)kv.Value);

        var args = new Dictionary<string, object?>();
        foreach (var pair in SplitRespectingQuotes(text))
        {
            var eq = pair.IndexOf('=');
            if (eq <= 0) throw new FormatException($"Expected name=value but got '{pair}'.");
            var value = pair[(eq + 1)..].Trim('"');
            args[pair[..eq]] = int.TryParse(value, out var i) ? i
                             : double.TryParse(value, System.Globalization.CultureInfo.InvariantCulture, out var d) ? d
                             : bool.TryParse(value, out var b) ? b
                             : value;
        }
        return args;
    }

    /// <summary>Splits on spaces, keeping "quoted text" together: topic="linear algebra".</summary>
    private static IEnumerable<string> SplitRespectingQuotes(string text)
    {
        var current = new System.Text.StringBuilder();
        var quoted = false;
        foreach (var c in text)
        {
            if (c == '"') quoted = !quoted;
            if (c == ' ' && !quoted)
            {
                if (current.Length > 0) yield return current.ToString();
                current.Clear();
            }
            else current.Append(c);
        }
        if (current.Length > 0) yield return current.ToString();
    }

    private static (string Head, string Tail) Split(string text)
    {
        var space = text.IndexOf(' ');
        return space < 0 ? (text, "") : (text[..space], text[(space + 1)..].Trim());
    }

    private static string Parameters(JsonElement schema)
    {
        if (!schema.TryGetProperty("properties", out var props)) return "";
        var required = schema.TryGetProperty("required", out var r) ? r.EnumerateArray().Select(x => x.GetString()).ToHashSet() : [];
        return string.Join(", ", props.EnumerateObject().Select(p =>
            $"{p.Name}{(required.Contains(p.Name) ? "" : "?")}: {(p.Value.TryGetProperty("type", out var t) ? t.ToString().Replace("\"", "") : "any")}"));
    }

    private static string Flags(Tool tool) =>
        tool.Annotations?.ReadOnlyHint == true ? "  [read-only]" : "  [CHANGES DATA]";

    private static string PrettyIfJson(string text)
    {
        try { return JsonSerializer.Serialize(JsonDocument.Parse(text), Pretty); }
        catch (JsonException) { return text; }
    }

    private static void WriteColored(ConsoleColor color, string text)
    {
        Console.ForegroundColor = color;
        Console.WriteLine(text);
        Console.ResetColor();
    }
}
