# Testing and debugging the Student MCP server

Run all commands from the `dotnet/` folder. Start at the top of this table and go further down
only when you need to:

| # | Way | Use it to... | Needs |
|---|---|---|---|
| 1 | [Automated tests](#1-automated-tests) | check nothing is broken, after every change | nothing |
| 2 | [Interactive console](#2-interactive-console) | call any tool by hand and see the exact result | nothing |
| 3 | [Breakpoints (F5)](#3-step-through-the-code-with-breakpoints) | step through a tool line by line | VS Code or Visual Studio |
| 4 | [Debug inside Claude](#4-debug-while-claude-is-using-the-server) | see what happens when Claude Desktop / Claude Code calls it | Claude Desktop or Claude Code |
| 5 | [MCP Inspector](#5-optional-mcp-inspector-web-ui) | click through tools in a web page | Node.js |

When something goes wrong, check [Common problems](#common-problems) first.

---

## 1. Automated tests

```bash
dotnet test                                                     # everything (48 tests)
dotnet test tests/StudentMcpServer.Tests                         # only the MCP server (22 tests)
dotnet test --filter "FullyQualifiedName~ListsAndReadsStudentFiles"   # one test by name
```

**How they work.** Each test starts the real server in memory and calls it through the real
MCP client, the same protocol Claude uses. Every test gets a fresh copy of the database, and a
fake Open Library (`FakeApi`) answers instead of the internet, so the tests are fast and can't be
broken by a network problem. See `tests/StudentMcpServer.Tests/TestServer.cs`.

**Adding a test** is a few lines. Copy an existing one in `StudentMcpServerTests.cs`:

```csharp
[Fact]
public async Task MeiHasNoGpaYet()
{
    await using var server = await TestServer.StartAsync();
    var transcript = Json(await server.CallAsync("get_student_transcript", ("studentId", "S1003")));
    Assert.Equal(JsonValueKind.Null, transcript.GetProperty("gpa").ValueKind);
}
```

To test how a tool handles the external API failing, start the server with
`TestServer.StartAsync(FakeApi.Failing(HttpStatusCode.ServiceUnavailable))`.

**Debugging a failing test:** in VS Code (with the C# Dev Kit extension), open the Testing
panel, right-click the test and choose **Debug Test**. In Visual Studio, open Test Explorer,
right-click and choose **Debug**. Breakpoints in the server's code are hit, because the server
runs inside the test.

---

## 2. Interactive console

The quickest way to try a tool by hand: no AI, no configuration.

```bash
dotnet run --project src/StudentMcpServer -- --console
```

```
mcp> tools
mcp> call get_student studentId=S1001
mcp> call search_students major=Computer limit=5
mcp> call find_textbooks topic="linear algebra" limit=3
mcp> call find_textbooks {"topic":"calculus","limit":3}         # JSON works too
mcp> call read_student_file studentId=S1001 fileName=advisor-notes.md
mcp> read student://S1001/transcript
mcp> prompt student_progress_report studentId=S1001
mcp> exit
```

**How to read the output:**
- `OK (12 ms)` followed by the result: this is exactly what the AI would receive.
- `Tool returned an error`: the error message the AI would see.
- Lines starting `warn:` / `fail:` come from the server's own logging. A `fail: ...threw an
  unhandled exception` line with a stack trace appears for every tool error, even expected ones
  like "No student with id". The stack trace shows the file and line that threw.
- `tools` marks each tool `[read-only]` or `[CHANGES DATA]`.

**Useful options:**
- `--verbose`: also show every MCP request and response.
- `--database scratch.db`: experiment without touching the normal sample database, for example
  when trying `enroll_student`.
- You can also pipe in a list of commands, which is handy for repeating a check:
  `printf 'call get_student studentId=S1001\nexit\n' | dotnet run --project src/StudentMcpServer -- --console`

The console runs the server inside the same process and talks to it through the real MCP client,
so what you see is what an AI app gets. It's also why breakpoints work (next section).

---

## 3. Step through the code with breakpoints

**VS Code** (install the **C# Dev Kit** extension):
1. Open the repository folder (`RAG-1`) in VS Code.
2. Put a breakpoint in a tool, e.g. `GetStudentTranscript` in `Tools/StudentDatabaseTools.cs`, or
   in `StudentDatabase.GetTranscript`.
3. Open **Run and Debug** (Ctrl+Shift+D), choose **Student MCP: interactive console**, press **F5**.
4. In the terminal, type `call get_student_transcript studentId=S1001`. The debugger stops on
   your breakpoint, and you can inspect variables and step with F10 / F11.

The configurations live in `.vscode/launch.json`. **Student MCP: agent demo** debugs the
`students` command instead (an agent using the server).

**Visual Studio:**
1. Open `dotnet/RagDotNet.slnx`.
2. Right-click **StudentMcpServer** → **Set as Startup Project**.
3. In the toolbar's run dropdown, choose the **Interactive console** profile.
4. Set a breakpoint and press **F5**, then type commands in the console window.

The profiles live in `src/StudentMcpServer/Properties/launchSettings.json`.

---

## 4. Debug while Claude is using the server

When Claude Desktop or Claude Code starts the server, you can't press F5. Instead, make the
server wait for you:

1. Build: `dotnet build src/StudentMcpServer`
2. Add `--wait-for-debugger` to the server's arguments in your Claude config (see the
   [README](README.md#connect-it-to-claude-desktop-or-claude-code)):
   ```json
   "args": ["<path>/StudentMcpServer.dll", "--wait-for-debugger"]
   ```
3. Restart Claude Desktop (or start Claude Code). The server starts and pauses.
4. Attach the debugger:
   - **VS Code:** Run and Debug → **Attach to Student MCP server** → pick the
     `StudentMcpServer` process.
   - **Visual Studio:** Debug → **Attach to Process** → pick `StudentMcpServer` (or the `dotnet`
     process running `StudentMcpServer.dll`).
5. Set breakpoints and ask Claude something that uses the tools.
6. **Remove `--wait-for-debugger` afterwards**, or the server will wait forever next time.

**Reading the server's logs.** Everything the server logs goes to stderr, and Claude saves it:
- **Claude Desktop:** `mcp-server-student-mcp.log` in the logs folder.
  - macOS: `~/Library/Logs/Claude/`
  - Windows: `%APPDATA%\Claude\logs\`
- **Claude Code:** `claude mcp list` shows whether the server connected, `/mcp` inside a
  session shows its status and tools, and `claude --debug` prints MCP errors.

Add `--verbose` to the args for much more detail, including every MCP message.

**Testing with a real AI, without Claude Desktop:** the agent demo starts the server and gives its
tools to a Claude agent:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
dotnet run --project src/RagApp -- students --live -q "How is Diego Martinez doing in calculus?"
```

It prints each tool call the AI made, which helps when the AI picks the wrong tool. If it does,
improve that tool's `[Description]`, which is all the AI knows about it.

---

## 5. Optional: MCP Inspector (web UI)

The [MCP Inspector](https://github.com/modelcontextprotocol/inspector) is the official
point-and-click tester for any MCP server. It needs [Node.js](https://nodejs.org); everything
above doesn't.

```bash
dotnet build src/StudentMcpServer
npx @modelcontextprotocol/inspector dotnet src/StudentMcpServer/bin/Debug/net10.0/StudentMcpServer.dll
```

It opens a browser page where you can list and call tools, read resources, try prompts and see
the raw messages.

---

## Common problems

| Symptom | Likely cause and fix |
|---|---|
| Claude says the server failed, disconnected, or sent invalid JSON | Something wrote to **stdout**, which is reserved for MCP messages. The usual culprits: (1) starting it with plain `dotnet run`, which prints "Using launch settings from ..." first. Use the built `.dll`, or add `--no-launch-profile`. (2) A `Console.WriteLine` in your code. Use `ILogger` instead, which goes to stderr. |
| A tool you added doesn't appear | Check that it has `[McpServerTool]` and its class is in the `WithTools([...])` list in `StudentServerSetup.cs`. Then rebuild **and restart** Claude, which reads the tool list once at startup. |
| Code changes have no effect in Claude | Claude runs the built `.dll`. Run `dotnet build`, then restart Claude Desktop (or the Claude Code session). |
| The AI sees only "An error occurred invoking 'x'" | An unexpected exception (a bug) happened; the SDK hides its details from the AI. The full stack trace is in the server log (or the console's `fail:` line). For errors the AI *should* see, throw `McpException("clear message")`. |
| Textbook tools fail | No internet, a firewall, or Open Library is down. The log shows `warn: ...TextbookApiClient` with the reason. `get_student_overview` still works without the books. |
| Want to reset the sample data | Delete `students.db` next to the program (`bin/Debug/net10.0/SampleData/`), or use `--database new.db`. |
| The server hangs at start | `--wait-for-debugger` is still in the args. |
| The AI calls the wrong tool or passes wrong values | Improve the tool's and parameters' `[Description]` text, which is all the AI knows about them. Check what it sees with `tools` in the console. |
