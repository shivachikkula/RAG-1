using System.Text.RegularExpressions;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using RagCore;

namespace RagTypes;

/// <summary>
/// Graph RAG: an extractor agent turns every chunk into (subject, relation, object) facts, building
/// a knowledge graph. For a question, entities it mentions are found in the graph, their neighbours
/// are followed, and the answer uses those facts plus the chunks they came from. Strong on
/// relationship and multi-hop questions ("who designed the software the flagship robot uses?").
/// </summary>
public sealed partial class GraphRag(LlmProvider llm, IEmbeddingGenerator<string, Embedding<float>> embedder, int hops = 1) : RagPipeline(llm, embedder)
{
    public sealed record Triple(string Subject, string Relation, string Object);
    public sealed record Triples(List<Triple> Items);

    private static readonly HashSet<string> NotEntities = ["The", "It", "Its", "Each", "Every", "If", "All", "To", "Customers", "Operators", "Damage", "Batteries"];

    /// <summary>entity -> (relation, other entity, chunk id). Relations stored in both directions.</summary>
    private readonly Dictionary<string, HashSet<(string Relation, string Other, string ChunkId)>> _edges = [];
    private readonly Dictionary<string, Document> _chunks = [];
    private VectorStore _store = null!;
    private ChatClientAgent _extractor = null!;
    private ChatClientAgent _answerer = null!;

    public override string Name => "graph";
    public override string Summary => "LLM-built knowledge graph + neighbourhood expansion.";

    [GeneratedRegex(@"\b[A-Z][a-zA-Z0-9-]*(?:\s+[A-Z0-9][a-zA-Z0-9-]*)*\b")]
    private static partial Regex ProperNoun();

    /// <summary>Offline fallback: proper nouns in the same sentence are linked as "mentioned_with".</summary>
    private static List<Triple> CooccurrenceTriples(string text)
    {
        var triples = new List<Triple>();
        foreach (var sentence in Regex.Split(text, @"(?<=[.!?])\s+"))
        {
            var entities = ProperNoun().Matches(sentence).Select(m => m.Value).Where(e => e.Length > 2 && !NotEntities.Contains(e)).Distinct().ToList();
            for (var i = 0; i < entities.Count; i++)
                for (var j = i + 1; j < entities.Count; j++)
                    triples.Add(new Triple(entities[i], "mentioned_with", entities[j]));
        }
        return triples;
    }

    private void AddEdge(string entity, string relation, string other, string chunkId)
    {
        if (!_edges.TryGetValue(entity, out var set))
            _edges[entity] = set = [];
        set.Add((relation, other, chunkId));
    }

    public IReadOnlyCollection<string> Entities => _edges.Keys;

    public override async Task IndexAsync(IReadOnlyList<Document> docs, CancellationToken ct = default)
    {
        _extractor = CreateAgent("triple-extractor",
            "Extract factual (subject, relation, object) triples from the text. Use canonical, fully-named entities " +
            "(e.g. 'Stratus S1', not 'it'). Use short snake_case relations.");
        _answerer = CreateAgent("graph-answerer",
            "Answer using the graph facts and passages. Follow relationships across multiple hops when needed. Cite passages as [n].");

        var chunks = Chunker.Sentences(docs);
        foreach (var chunk in chunks)
        {
            _chunks[chunk.Id] = chunk;
            var extracted = ((await AskForAsync<Triples>(_extractor, $"<text>\n{chunk.Text}\n</text>", ct)).Items ?? [])
                .Where(t => !string.IsNullOrWhiteSpace(t.Subject) && !string.IsNullOrWhiteSpace(t.Object)).ToList();
            foreach (var t in extracted.Count > 0 ? extracted : CooccurrenceTriples(chunk.Text))
            {
                AddEdge(t.Subject.Trim(), t.Relation.Trim(), t.Object.Trim(), chunk.Id);
                AddEdge(t.Object.Trim(), "inverse:" + t.Relation.Trim(), t.Subject.Trim(), chunk.Id);
            }
        }
        _store = new VectorStore(Embedder);  // used to find starting entities when no name matches
        await _store.AddAsync(chunks, ct: ct);
    }

    public override async Task<RagResult> QueryAsync(string question, CancellationToken ct = default)
    {
        // 1. Entities whose every word appears in the question, longest names first.
        var questionTokens = TextUtil.Tokenize(question).ToHashSet();
        var seeds = _edges.Keys
            .Select(e => (Entity: e, Tokens: TextUtil.Tokenize(e)))
            .Where(x => x.Tokens.Count > 0 && x.Tokens.All(questionTokens.Contains))
            .OrderByDescending(x => x.Tokens.Count).Select(x => x.Entity).ToList();
        if (seeds.Count == 0)  // otherwise start from entities in the best-matching chunks
        {
            var topIds = (await _store.SearchAsync(question, 2, ct)).Select(h => h.Doc.Id).ToHashSet();
            seeds = _edges.Where(e => e.Value.Any(r => topIds.Contains(r.ChunkId))).Select(e => e.Key).ToList();
        }

        // 2. Walk the graph outward from the seeds.
        var frontier = seeds.ToHashSet();
        var visited = new HashSet<string>();
        var facts = new HashSet<string>();
        var chunkIds = new List<string>();
        for (var hop = 0; hop <= hops; hop++)
        {
            var next = new HashSet<string>();
            foreach (var entity in frontier.Except(visited))
            {
                visited.Add(entity);
                foreach (var (relation, other, chunkId) in _edges.GetValueOrDefault(entity) ?? [])
                {
                    facts.Add(relation.StartsWith("inverse:") ? $"- {other} --{relation["inverse:".Length..]}--> {entity}" : $"- {entity} --{relation}--> {other}");
                    next.Add(other);
                    if (!chunkIds.Contains(chunkId)) chunkIds.Add(chunkId);
                }
            }
            frontier = next;
        }

        // 3. Answer from the facts and their source chunks.
        var docs = chunkIds.Take(TopK * 2).Select(id => _chunks[id]).ToList();
        var prompt = $"<context>\nKnowledge-graph facts:\n{string.Join("\n", facts.Order().Take(60))}\n\nSource passages:\n{Chunker.FormatContext(docs)}\n</context>\n\n<question>\n{question}\n</question>";
        var answer = await AskAsync(_answerer, prompt, ct: ct);
        return new RagResult(answer, docs, new Dictionary<string, object?> { ["seed_entities"] = seeds.Take(5).ToList(), ["facts_used"] = facts.Count });
    }
}
