// Student MCP server. AI apps (Claude Desktop, Claude Code, an Agent Framework agent...) start
// this program and talk to it over stdin/stdout using the Model Context Protocol.
//
//   dotnet run --project src/StudentMcpServer
//   dotnet run --project src/StudentMcpServer -- --database my.db --files ./my-files

using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using StudentMcpServer;

var builder = Host.CreateApplicationBuilder(args);

// stdout carries the MCP messages, so all logging must go to stderr instead.
builder.Logging.AddConsole(options => options.LogToStandardErrorThreshold = LogLevel.Trace);

builder.Services
    .AddStudentMcpServer(StudentMcpOptions.FromConfiguration(builder.Configuration))
    .WithStdioServerTransport();

await builder.Build().RunAsync();
