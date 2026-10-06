// Student MCP server. AI apps (Claude Desktop, Claude Code, an Agent Framework agent...) start
// this program and talk to it over stdin/stdout using the Model Context Protocol.
//
//   dotnet run --project src/StudentMcpServer -- --console            interactive console for testing
//   dotnet run --project src/StudentMcpServer -- --console --verbose  ...also show every MCP message
//   dotnet bin/Debug/net10.0/StudentMcpServer.dll                     normal MCP server (stdio), as Claude runs it
//   ... --wait-for-debugger                                           pause until a debugger attaches
//   ... --database my.db --files ./my-files                           use your own data
//
// For the stdio server, MCP clients should run the built .dll (or use `dotnet run --no-launch-profile`):
// plain `dotnet run` prints "Using launch settings..." to stdout, which breaks the protocol.

using System.Diagnostics;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using StudentMcpServer;

// Our own on/off switches; everything else is passed to the configuration system.
string[] switches = ["--console", "--verbose", "--wait-for-debugger"];
var consoleMode = args.Contains("--console");
var verbose = args.Contains("--verbose");
var waitForDebugger = args.Contains("--wait-for-debugger");
args = [.. args.Where(a => !switches.Contains(a))];

if (waitForDebugger)
{
    // When Claude Desktop starts the server you can't press F5, so attach instead:
    // Visual Studio: Debug > Attach to Process; VS Code: the "Attach to Student MCP server" configuration.
    Console.Error.WriteLine($"[student-mcp] Waiting for a debugger to attach to process {Environment.ProcessId} ({Environment.ProcessPath})...");
    while (!Debugger.IsAttached)
        await Task.Delay(250);
    Console.Error.WriteLine("[student-mcp] Debugger attached.");
}

var options = StudentMcpOptions.FromConfiguration(new ConfigurationBuilder().AddEnvironmentVariables().AddCommandLine(args).Build());

if (consoleMode)
    return await McpConsole.RunAsync(options, verbose ? LogLevel.Trace : LogLevel.Warning);

var builder = Host.CreateApplicationBuilder(args);

// stdout carries the MCP messages, so all logging must go to stderr instead.
builder.Logging.AddConsole(o => o.LogToStandardErrorThreshold = LogLevel.Trace);
if (verbose)
    builder.Logging.SetMinimumLevel(LogLevel.Trace);

builder.Services
    .AddStudentMcpServer(options)
    .WithStdioServerTransport();

await builder.Build().RunAsync();
return 0;
