using System.IO.Pipelines;
using System.Net;
using System.Text;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using ModelContextProtocol.Client;
using ModelContextProtocol.Protocol;

namespace StudentMcpServer.Tests;

/// <summary>
/// Runs the real MCP server in memory, connected to a real MCP client through two pipes, with a
/// fresh database per test and a fake Open Library API (no internet needed).
/// </summary>
public sealed class TestServer : IAsyncDisposable
{
    private readonly IHost _host;
    private readonly string _tempDir;

    private TestServer(IHost host, McpClient client, FakeApi api, string tempDir)
    {
        _host = host;
        Client = client;
        Api = api;
        _tempDir = tempDir;
    }

    public McpClient Client { get; }
    public FakeApi Api { get; }

    public static async Task<TestServer> StartAsync(FakeApi? api = null)
    {
        api ??= FakeApi.Returning(OpenLibraryJson(("Sample Book", "Jane Author")));
        var tempDir = Directory.CreateTempSubdirectory("student-mcp-").FullName;
        var options = new StudentMcpOptions
        {
            DatabasePath = Path.Combine(tempDir, "students.db"),
            FilesPath = Path.Combine(AppContext.BaseDirectory, "SampleData", "files"),
        };

        var clientToServer = new Pipe();
        var serverToClient = new Pipe();
        var builder = Host.CreateApplicationBuilder();
        builder.Logging.ClearProviders();
        builder.Services
            .AddStudentMcpServer(options, http => http.ConfigurePrimaryHttpMessageHandler(() => api))
            .WithStreamServerTransport(clientToServer.Reader.AsStream(), serverToClient.Writer.AsStream());
        var host = builder.Build();
        await host.StartAsync();

        var client = await McpClient.CreateAsync(new StreamClientTransport(clientToServer.Writer.AsStream(), serverToClient.Reader.AsStream()));
        return new TestServer(host, client, api, tempDir);
    }

    public async Task<CallToolResult> CallAsync(string tool, params (string Name, object? Value)[] args) =>
        await Client.CallToolAsync(tool, args.ToDictionary(a => a.Name, a => a.Value));

    public static string TextOf(CallToolResult result) => string.Concat(result.Content.OfType<TextContentBlock>().Select(b => b.Text));

    public static string OpenLibraryJson(params (string Title, string Author)[] books) =>
        "{\"docs\":[" + string.Join(",", books.Select((b, i) =>
            $$"""{"key":"/works/OL{{i}}W","title":"{{b.Title}}","author_name":["{{b.Author}}"],"first_publish_year":2001,"isbn":["97800000000{{i}}"]}""")) + "]}";

    public async ValueTask DisposeAsync()
    {
        await Client.DisposeAsync();
        await _host.StopAsync();
        _host.Dispose();
        try { Directory.Delete(_tempDir, recursive: true); } catch (IOException) { }
    }
}

/// <summary>Stands in for the Open Library API and records the requests it receives.</summary>
public sealed class FakeApi(Func<HttpRequestMessage, HttpResponseMessage> respond) : HttpMessageHandler
{
    public List<HttpRequestMessage> Requests { get; } = [];

    public static FakeApi Returning(string json) =>
        new(_ => new HttpResponseMessage(HttpStatusCode.OK) { Content = new StringContent(json, Encoding.UTF8, "application/json") });

    public static FakeApi Failing(HttpStatusCode status) => new(_ => new HttpResponseMessage(status));

    protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
    {
        Requests.Add(request);
        return Task.FromResult(respond(request));
    }
}
