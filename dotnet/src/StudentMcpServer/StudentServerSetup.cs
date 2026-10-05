using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using StudentMcpServer.Data;
using StudentMcpServer.ExternalApi;
using StudentMcpServer.Prompts;
using StudentMcpServer.Resources;
using StudentMcpServer.Storage;
using StudentMcpServer.Tools;

namespace StudentMcpServer;

/// <summary>Where the server's data lives. Each value can be changed with a command-line option or environment variable.</summary>
public sealed class StudentMcpOptions
{
    public required string DatabasePath { get; init; }
    public required string FilesPath { get; init; }
    public string TextbookApiBaseUrl { get; init; } = "https://openlibrary.org/";

    /// <summary>Reads --database / --files / --textbook-api, or STUDENT_MCP_DATABASE / STUDENT_MCP_FILES / STUDENT_MCP_TEXTBOOK_API.</summary>
    public static StudentMcpOptions FromConfiguration(IConfiguration config)
    {
        var sampleData = Path.Combine(AppContext.BaseDirectory, "SampleData");
        return new StudentMcpOptions
        {
            DatabasePath = config["database"] ?? config["STUDENT_MCP_DATABASE"] ?? Path.Combine(sampleData, "students.db"),
            FilesPath = config["files"] ?? config["STUDENT_MCP_FILES"] ?? Path.Combine(sampleData, "files"),
            TextbookApiBaseUrl = config["textbook-api"] ?? config["STUDENT_MCP_TEXTBOOK_API"] ?? "https://openlibrary.org/",
        };
    }
}

public static class StudentServerSetup
{
    public const string Instructions =
        "Student information service. Use search_students to find a student's id, then the other tools with that id. " +
        "get_student_overview is the quickest way to see everything about one student. " +
        "enroll_student changes data: only use it when the user clearly asks to enroll someone.";

    /// <summary>
    /// Registers the data sources and the MCP server with its tools, resources and prompts.
    /// The caller picks the transport (stdio for Claude Desktop / Claude Code, or a stream in tests).
    /// </summary>
    public static IMcpServerBuilder AddStudentMcpServer(this IServiceCollection services, StudentMcpOptions options, Action<IHttpClientBuilder>? configureHttp = null)
    {
        services.AddSingleton(new StudentDatabase(options.DatabasePath));
        services.AddSingleton(sp => new StudentFileStore(options.FilesPath, sp.GetRequiredService<ILogger<StudentFileStore>>()));

        var http = services.AddHttpClient<TextbookApiClient>(client =>
        {
            client.BaseAddress = new Uri(options.TextbookApiBaseUrl);
            client.Timeout = TimeSpan.FromSeconds(15);
            client.DefaultRequestHeaders.UserAgent.ParseAdd("StudentMcpServer/1.0 (educational demo)");
        });
        configureHttp?.Invoke(http);

        return services
            .AddMcpServer(server =>
            {
                server.ServerInfo = new() { Name = "student-mcp", Version = "1.0.0" };
                server.ServerInstructions = Instructions;
            })
            .WithTools([typeof(StudentDatabaseTools), typeof(StudentFileTools), typeof(TextbookTools), typeof(StudentOverviewTools)])
            .WithResources([typeof(StudentResources)])
            .WithPrompts([typeof(StudentPrompts)]);
    }
}
