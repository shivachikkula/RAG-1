using Microsoft.Extensions.AI;

namespace RagCore;

public readonly record struct Hit(Document Doc, double Score);

/// <summary>In-memory vector store using cosine similarity (vectors are normalised).</summary>
public sealed class VectorStore(IEmbeddingGenerator<string, Embedding<float>> embedder)
{
    private readonly List<(Document Doc, float[] Vector)> _items = [];

    /// <summary>Adds documents. <paramref name="textsToEmbed"/> lets you embed something other than
    /// the document text, e.g. a chunk with LLM-written context prepended.</summary>
    public async Task AddAsync(IReadOnlyList<Document> docs, IReadOnlyList<string>? textsToEmbed = null, CancellationToken ct = default)
    {
        if (docs.Count == 0) return;
        var embeddings = await embedder.GenerateAsync(textsToEmbed ?? docs.Select(d => d.Text).ToList(), cancellationToken: ct);
        for (var i = 0; i < docs.Count; i++)
            _items.Add((docs[i], embeddings[i].Vector.ToArray()));
    }

    public async Task<List<Hit>> SearchAsync(string query, int k = 4, CancellationToken ct = default) =>
        SearchByVector(await EmbedAsync(query, ct), k);

    public async Task<float[]> EmbedAsync(string text, CancellationToken ct = default) =>
        (await embedder.GenerateAsync([text], cancellationToken: ct))[0].Vector.ToArray();

    public List<Hit> SearchByVector(float[] vector, int k = 4) =>
        _items.Select(item => new Hit(item.Doc, Dot(item.Vector, vector)))
              .OrderByDescending(h => h.Score)
              .Take(k)
              .ToList();

    private static double Dot(float[] a, float[] b)
    {
        double sum = 0;
        for (var i = 0; i < a.Length; i++) sum += a[i] * b[i];
        return sum;
    }
}

/// <summary>Okapi BM25 keyword search: strong on exact terms like product codes and IDs.</summary>
public sealed class Bm25Index(double k1 = 1.5, double b = 0.75)
{
    private readonly List<(Document Doc, Dictionary<string, int> TermFreq, int Length)> _docs = [];
    private readonly Dictionary<string, int> _docFreq = [];
    private double _avgLength;

    public void Add(IReadOnlyList<Document> docs, IReadOnlyList<string>? textsToIndex = null)
    {
        for (var i = 0; i < docs.Count; i++)
        {
            var tokens = TextUtil.Tokenize(textsToIndex?[i] ?? docs[i].Text);
            var tf = tokens.GroupBy(t => t).ToDictionary(g => g.Key, g => g.Count());
            _docs.Add((docs[i], tf, tokens.Count));
            foreach (var term in tf.Keys)
                _docFreq[term] = _docFreq.GetValueOrDefault(term) + 1;
        }
        _avgLength = _docs.Average(d => (double)d.Length);
    }

    public List<Hit> Search(string query, int k = 4)
    {
        var n = _docs.Count;
        var terms = TextUtil.Tokenize(query);
        return _docs.Select(d =>
            {
                double score = 0;
                foreach (var term in terms)
                {
                    if (!d.TermFreq.TryGetValue(term, out var freq)) continue;
                    var df = _docFreq[term];
                    var idf = Math.Log(1 + (n - df + 0.5) / (df + 0.5));
                    score += idf * freq * (k1 + 1) / (freq + k1 * (1 - b + b * d.Length / _avgLength));
                }
                return new Hit(d.Doc, score);
            })
            .Where(h => h.Score > 0)
            .OrderByDescending(h => h.Score)
            .Take(k)
            .ToList();
    }
}

public static class RankFusion
{
    /// <summary>Reciprocal Rank Fusion: score(doc) = sum over lists of 1 / (k + rank).</summary>
    public static List<Hit> Reciprocal(IEnumerable<IReadOnlyList<Hit>> resultLists, int topN = 4, int k = 60)
    {
        var scores = new Dictionary<string, (Document Doc, double Score)>();
        foreach (var results in resultLists)
            for (var rank = 0; rank < results.Count; rank++)
            {
                var doc = results[rank].Doc;
                var previous = scores.GetValueOrDefault(doc.Id, (doc, 0));
                scores[doc.Id] = (doc, previous.Score + 1.0 / (k + rank + 1));
            }
        return scores.Values.OrderByDescending(s => s.Score).Take(topN).Select(s => new Hit(s.Doc, s.Score)).ToList();
    }
}
