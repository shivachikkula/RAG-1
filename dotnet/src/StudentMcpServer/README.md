# Student MCP Server

An **MCP server** that gives AI apps access to student information from three kinds of source:
a **database**, **file storage**, and an **external internet API**. It's built with the official
[Model Context Protocol C# SDK](https://github.com/modelcontextprotocol/csharp-sdk).

## What is MCP?

The **Model Context Protocol** is a standard way for AI apps (Claude Desktop, Claude Code,
VS Code, your own agents) to use outside tools and data. You write a server once, and any MCP
app can use it. The app starts this program and they talk over stdin/stdout. The AI reads each
tool's description and decides when to call it.

```
 Claude Desktop / Claude Code / Agent Framework agent
                │  MCP (JSON messages over stdin/stdout)
                ▼
        ┌──────────────────────┐
        │  Student MCP Server  │
        └──┬────────┬───────┬──┘
           ▼        ▼       ▼
       SQLite   student   Open Library
      database   files     web API
```

## What it offers

**Tools** (actions the AI can call):

| Source | Tool | What it does |
|---|---|---|
| Database | `search_students` | find students by name and/or major |
| Database | `get_student` | one student's profile |
| Database | `get_student_transcript` | courses, grades, GPA and credits |
| Database | `list_courses` | the course catalog |
| Database | `enroll_student` | enroll a student in a course (**changes data**) |
| File storage | `list_student_files` | a student's documents |
| File storage | `read_student_file` | read one document |
| External API | `find_textbooks` | search Open Library for books on a topic |
| Database + API | `recommend_textbooks_for_course` | look up a course's topic, then search for books |
| All three | `get_student_overview` | profile, GPA, documents and suggested reading in one call |

**Resources** (data an app can attach directly, like a file): `student://{studentId}/transcript`
and `student-files://{studentId}/{fileName}`.

**Prompts** (ready-made requests a user can pick from a menu): `student_progress_report` and
`study_plan`.

Every tool is marked read-only or not (`enroll_student` is the only one that changes data), so
apps can ask the user before running it.

## The sample data

- **Database:** `SampleData/students.db` is created and filled automatically the first time the
  server runs: 5 students, 6 courses, and their grades. Delete the file to reset it.
- **File storage:** `SampleData/files/<student id>/` holds advisor notes, a lab report, a
  welcome letter and similar documents.
- **External API:** [Open Library](https://openlibrary.org/developers/api) search. It's free and
  needs no key, but it does need internet access.

| Id | Student | Major |
|---|---|---|
| S1001 | Aisha Khan | Computer Science |
| S1002 | Diego Martinez | Biology |
| S1003 | Mei Chen | Mathematics |
| S1004 | Liam O'Brien | History |
| S1005 | Priya Patel | Computer Science |

## Try it with an AI agent (no setup)

From the `dotnet/` folder:

```bash
dotnet run --project src/RagApp -- students                 # offline: connects and calls one tool
dotnet run --project src/RagApp -- students --live -q "How is Aisha Khan doing and what should she focus on?"
```

With `--live` (and `ANTHROPIC_API_KEY` set), a Microsoft Agent Framework agent receives the
server's tools and Claude decides which to call. The code is in `src/RagApp/StudentAgentDemo.cs`,
and the key lines are:

```csharp
var tools = await mcpClient.ListToolsAsync();     // MCP tools...
options.Tools = [.. tools];                        // ...are ordinary agent tools
var agent = chatClient.AsAIAgent(new ChatClientAgentOptions { ChatOptions = options });
```

## Connect it to Claude Desktop or Claude Code

First build it, from the `dotnet/` folder:

```bash
dotnet build src/StudentMcpServer
```

The program is now at `dotnet/src/StudentMcpServer/bin/Debug/net10.0/StudentMcpServer.dll`. In
the steps below, replace `<path>` with the full path to that file.

**Claude Desktop:** open Settings → Developer → Edit Config, and add:

```json
{
  "mcpServers": {
    "student-mcp": {
      "command": "dotnet",
      "args": ["<path>"]
    }
  }
}
```

Restart Claude Desktop, then ask something like *"Use the student tools: how is Diego Martinez
doing in calculus?"*

**Claude Code:**

```bash
claude mcp add student-mcp -- dotnet <path>
```

## Settings

| Option | Environment variable | Default |
|---|---|---|
| `--database <file>` | `STUDENT_MCP_DATABASE` | `SampleData/students.db` next to the program |
| `--files <folder>` | `STUDENT_MCP_FILES` | `SampleData/files` next to the program |
| `--textbook-api <url>` | `STUDENT_MCP_TEXTBOOK_API` | `https://openlibrary.org/` |

## How the code is organised

```
StudentMcpServer/
├── Program.cs                     starts the server on stdin/stdout
├── StudentServerSetup.cs          registers data sources, tools, resources, prompts
├── Data/StudentDatabase.cs        SQLite: tables, sample data, queries, GPA
├── Storage/StudentFileStore.cs    reads student folders safely
├── ExternalApi/TextbookApiClient.cs   calls the Open Library API
├── Tools/                         the 10 MCP tools
├── Resources/StudentResources.cs  the 2 resources
├── Prompts/StudentPrompts.cs      the 2 prompts
└── SampleData/files/              the sample documents
```

A tool is just a C# method with attributes. The `[Description]` text is what the AI reads to
decide when to use it:

```csharp
[McpServerTool(Name = "get_student", ReadOnly = true)]
[Description("Get one student's profile by student id (e.g. S1001).")]
public static Student GetStudent(StudentDatabase db, [Description("Student id, e.g. S1001")] string studentId) =>
    db.GetStudent(studentId) ?? throw new McpException($"No student with id '{studentId}'.");
```

`StudentDatabase db` isn't something the AI fills in. The SDK supplies it automatically
(dependency injection), so the AI only sees `studentId`.

## Safety built in

- **SQL injection:** the AI's input is always passed as query parameters, never pasted into
  SQL. A test checks this.
- **File access:** a tool can only read files inside that student's folder. `../` tricks,
  absolute paths, other file types and files over 256 KB are refused. Tests check each case.
- **Errors:** a missing student, a missing file or an API outage comes back to the AI as a clear
  message it can act on. `get_student_overview` still works when the API is down, and just
  leaves out the book suggestions.
- **Logging:** logs go to stderr, because stdout carries the MCP messages.

## Making it real

The three sources sit behind small classes, so you can swap each one without touching the tools:
- **Database:** replace SQLite in `StudentDatabase` with SQL Server or PostgreSQL. Only the
  connection and SQL dialect change.
- **File storage:** point `StudentFileStore` at Azure Blob Storage or S3, with one folder per student.
- **External API:** `TextbookApiClient` shows the pattern (typed `HttpClient`, timeout, error
  handling) for calling any web API.

## Tests

`tests/StudentMcpServer.Tests` (22 tests) runs the real server in memory and calls it through a
real MCP client, with a fresh database per test and a fake Open Library, so no internet is needed.
One test runs the full loop: an agent gets the MCP tools, a fake Claude asks to call one, and the
database result goes back to Claude.

```bash
dotnet test
```
