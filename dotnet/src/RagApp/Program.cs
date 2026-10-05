// RAG runner. Examples (run from the dotnet folder):
//   dotnet run --project src/RagApp -- list
//   dotnet run --project src/RagApp -- hybrid -q "Which ISO standard do the robots comply with?"
//   dotnet run --project src/RagApp -- naive hybrid graph -q "Who designed Cirrus?"
//   dotnet run --project src/RagApp -- all --mock --report report.md
//   dotnet run --project src/RagApp -- chat
//   dotnet run --project src/RagApp -- students --live -q "How is Aisha Khan doing?"

using System.Diagnostics;
using System.Text;
using RagCore;
using RagTypes;

var types = new List<string>();
string? question = null;
string? dataDir = null, reportPath = null;
var mode = LlmMode.Auto;

for (var i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "-q" or "--question": question = args[++i]; break;
        case "--data": dataDir = args[++i]; break;
        case "--report": reportPath = args[++i]; break;
        case "--mock": mode = LlmMode.Mock; break;
        case "--live": mode = LlmMode.Live; break;
        case "-h" or "--help": PrintHelp(); return 0;
        default: types.Add(args[i]); break;
    }
}

if (types.Count == 0 || types is ["list"])
{
    var listing = LlmProvider.CreateMock();
    foreach (var name in RagRegistry.All.Keys)
        Console.WriteLine($"{name,-22} {RagRegistry.Create(name, listing).Summary}");
    return 0;
}

LlmProvider llm;
try
{
    llm = LlmProvider.Create(mode);
}
catch (InvalidOperationException ex)
{
    Console.Error.WriteLine($"error: {ex.Message}");
    return 2;
}

var docs = DocumentLoader.LoadDirectory(dataDir ?? DocumentLoader.FindSampleData());

if (types is ["chat"])
    return await ChatAsync();
if (types is ["students"])
    return await StudentAgentDemo.RunAsync(llm, question ?? StudentAgentDemo.DefaultQuestion);
question ??= "How long does the Stratus S1 battery last?";

if (types is ["all"])
    types = [.. RagRegistry.All.Keys];
var unknown = types.Where(t => !RagRegistry.All.ContainsKey(t)).ToList();
if (unknown.Count > 0)
{
    Console.Error.WriteLine($"error: unknown RAG type(s): {string.Join(", ", unknown)}. Run 'list' to see them.");
    return 2;
}

var rows = new List<(string Name, string Status, double Seconds, RagResult? Result)>();
foreach (var name in types)
{
    var timer = Stopwatch.StartNew();
    Console.WriteLine($"\n{new string('=', 70)}\n# {name}  (llm: {llm.ModelName})\nQ: {question}\n");
    try
    {
        var rag = RagRegistry.Create(name, llm);
        await rag.IndexAsync(docs);
        var result = await rag.QueryAsync(question);
        Console.WriteLine(result.Pretty());
        rows.Add((name, "ok", timer.Elapsed.TotalSeconds, result));
    }
    catch (Exception ex)  // keep running the other types; report the failure at the end
    {
        Console.Error.WriteLine($"FAILED: {ex.GetType().Name}: {ex.Message}");
        rows.Add((name, $"failed: {ex.GetType().Name}", timer.Elapsed.TotalSeconds, null));
    }
}

if (reportPath is not null)
    File.WriteAllText(reportPath, BuildReport());
return rows.Any(r => r.Result is null) ? 1 : 0;

async Task<int> ChatAsync()
{
    var rag = new ConversationalRag(llm, new HashingEmbeddingGenerator());
    await rag.IndexAsync(docs);
    Console.WriteLine($"# conversational chat (llm: {llm.ModelName}). Type 'reset' to clear history, 'exit' to quit.");
    while (true)
    {
        Console.Write("\nYou: ");
        var line = Console.ReadLine()?.Trim();
        if (line is null or "exit" or "quit") return 0;
        if (line == "reset") { rag.Reset(); continue; }
        if (line.Length > 0) Console.WriteLine(new string('-', 40) + "\n" + (await rag.QueryAsync(line)).Pretty());
    }
}

string BuildReport()
{
    var sb = new StringBuilder($"## RAG run ({llm.ModelName}, .NET)\n\n**Question:** {question}\n\n| Type | Status | Time (s) | Sources |\n|---|---|---|---|\n");
    foreach (var (name, status, seconds, result) in rows)
        sb.AppendLine($"| {name} | {status} | {seconds:F2} | {string.Join(", ", result?.Sources.Select(d => d.Id) ?? [])} |");
    foreach (var (name, _, _, result) in rows.Where(r => r.Result is not null))
        sb.AppendLine($"\n### {name}\n\n> {result!.Answer.Replace("\n", "\n> ")}");
    return sb.ToString();
}

static void PrintHelp() => Console.WriteLine("""
    Usage: dotnet run --project src/RagApp -- [list | chat | all | <type> ...] [options]

      list               show the 14 RAG types
      chat               interactive conversational RAG
      students           an agent using the Student MCP server's tools
      all | <type> ...   run one or more RAG types on a question

    Options:
      -q, --question     question to ask
      --data <folder>    folder of .md/.txt files (default: data/sample_docs)
      --mock             use the offline mock LLM (no API key needed)
      --live             require Claude (needs ANTHROPIC_API_KEY)
      --report <file>    write a Markdown summary
    """);
