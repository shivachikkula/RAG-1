using RagCore;
using RagTypes;

namespace RagTests;

/// <summary>Offline tests: every RAG type indexes the sample docs and answers using the mock LLM.</summary>
public class PipelineTests
{
    private static readonly List<Document> Docs = DocumentLoader.LoadDirectory(DocumentLoader.FindSampleData());

    public static TheoryData<string> AllTypes() => [.. RagRegistry.All.Keys];

    [Theory]
    [MemberData(nameof(AllTypes))]
    public async Task EveryTypeAnswersWithSources(string name)
    {
        var rag = RagRegistry.Create(name, LlmProvider.CreateMock());
        await rag.IndexAsync(Docs);
        var result = await rag.QueryAsync("How long does the Stratus S1 battery last on a single charge?");
        Assert.False(string.IsNullOrWhiteSpace(result.Answer));
        Assert.NotEmpty(result.Sources);
    }

    [Fact]
    public async Task DenseRetrievalFindsBatteryChunk()
    {
        var store = new VectorStore(new HashingEmbeddingGenerator());
        await store.AddAsync(Chunker.Sentences(Docs));
        var top = (await store.SearchAsync("Stratus S1 battery hours single charge", 1))[0];
        Assert.Contains("10 hours", top.Doc.Text);
    }

    [Fact]
    public void Bm25FindsExactTerm()
    {
        var index = new Bm25Index();
        index.Add(Chunker.Sentences(Docs));
        Assert.Contains("3691", index.Search("ISO 3691-4", 1)[0].Doc.Text);
    }

    [Fact]
    public void RankFusionPrefersDocsInBothLists()
    {
        var chunks = Chunker.Sentences(Docs);
        var fused = RankFusion.Reciprocal([[new Hit(chunks[0], 1), new Hit(chunks[1], 0.5)], [new Hit(chunks[1], 1)]], 2);
        Assert.Equal(chunks[1].Id, fused[0].Doc.Id);
    }

    [Fact]
    public async Task ConversationalKeepsHistory()
    {
        var rag = new ConversationalRag(LlmProvider.CreateMock(), new HashingEmbeddingGenerator());
        await rag.IndexAsync(Docs);
        await rag.QueryAsync("Tell me about the Cumulus C2");
        var result = await rag.QueryAsync("What is its battery life?");
        Assert.Equal(2, result.Details["turn"]);
    }

    [Fact]
    public async Task GraphBuildsEntitiesOffline()
    {
        var rag = new GraphRag(LlmProvider.CreateMock(), new HashingEmbeddingGenerator());
        await rag.IndexAsync(Docs);
        Assert.Contains("Tomas Lindqvist", rag.Entities);
    }

    [Fact]
    public async Task AgenticRunsTheToolLoop()
    {
        var rag = new AgenticRag(LlmProvider.CreateMock(), new HashingEmbeddingGenerator());
        await rag.IndexAsync(Docs);
        var result = await rag.QueryAsync("How long does the Stratus S1 battery last?");
        var calls = Assert.IsType<List<string>>(result.Details["tool_calls"]);
        Assert.Contains(calls, c => c.StartsWith("search_knowledge_base"));
    }
}
