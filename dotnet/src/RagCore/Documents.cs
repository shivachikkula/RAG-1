using System.Text.RegularExpressions;

namespace RagCore;

/// <summary>A piece of text plus where it came from. Chunks are Documents too.</summary>
public sealed class Document
{
    public required string Id { get; init; }
    public required string Source { get; init; }
    public required string Text { get; init; }

    /// <summary>For chunks: the Id of the document or section they were cut from.</summary>
    public string? ParentId { get; init; }

    /// <summary>Extra per-document data, e.g. the LLM-written context in Contextual Retrieval.</summary>
    public Dictionary<string, string> Metadata { get; init; } = [];

    /// <summary>First line of the text without Markdown heading marks.</summary>
    public string Title => Text.Split('\n')[0].TrimStart('#', ' ');
}

public static class DocumentLoader
{
    /// <summary>Loads every .md and .txt file in a folder as one Document.</summary>
    public static List<Document> LoadDirectory(string path) =>
        Directory.EnumerateFiles(path)
            .Where(f => f.EndsWith(".md") || f.EndsWith(".txt"))
            .Order(StringComparer.Ordinal)
            .Select(f => new Document { Id = Path.GetFileNameWithoutExtension(f), Source = Path.GetFileName(f), Text = File.ReadAllText(f) })
            .ToList();

    /// <summary>Finds the repo's data/sample_docs folder by walking up from the running program.</summary>
    public static string FindSampleData()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            var candidate = Path.Combine(dir.FullName, "data", "sample_docs");
            if (Directory.Exists(candidate))
                return candidate;
        }
        throw new DirectoryNotFoundException("Could not find data/sample_docs. Pass --data <folder>.");
    }
}

public static partial class Chunker
{
    [GeneratedRegex(@"(?<=[.!?])\s+|\n{2,}")]
    private static partial Regex SentenceBoundary();

    public static List<string> SplitSentences(string text) =>
        SentenceBoundary().Split(text).Select(s => s.Trim()).Where(s => s.Length > 0).ToList();

    /// <summary>Naive fixed-size character chunks (the baseline strategy).</summary>
    public static List<Document> Fixed(IEnumerable<Document> docs, int chunkSize = 500, int overlap = 0)
    {
        var chunks = new List<Document>();
        var step = Math.Max(1, chunkSize - overlap);
        foreach (var doc in docs)
        {
            for (int start = 0, n = 0; start < doc.Text.Length; start += step, n++)
            {
                var piece = doc.Text.Substring(start, Math.Min(chunkSize, doc.Text.Length - start)).Trim();
                if (piece.Length > 0)
                    chunks.Add(MakeChunk(doc, piece, n));
            }
        }
        return chunks;
    }

    /// <summary>Packs whole sentences up to maxChars, repeating the last sentences for overlap.</summary>
    public static List<Document> Sentences(IEnumerable<Document> docs, int maxChars = 400, int overlapSentences = 1)
    {
        var chunks = new List<Document>();
        foreach (var doc in docs)
        {
            var current = new List<string>();
            var n = 0;
            foreach (var sentence in SplitSentences(doc.Text))
            {
                if (current.Count > 0 && current.Sum(s => s.Length + 1) + sentence.Length > maxChars)
                {
                    chunks.Add(MakeChunk(doc, string.Join(' ', current), n++));
                    current = overlapSentences > 0 ? current.TakeLast(overlapSentences).ToList() : [];
                }
                current.Add(sentence);
            }
            if (current.Count > 0)
                chunks.Add(MakeChunk(doc, string.Join(' ', current), n));
        }
        return chunks;
    }

    private static Document MakeChunk(Document doc, string text, int n) =>
        new() { Id = $"{doc.Id}#{n}", Source = doc.Source, Text = text, ParentId = doc.Id };

    /// <summary>Numbers the chunks so the model can cite them as [1], [2], ...</summary>
    public static string FormatContext(IEnumerable<Document> docs) =>
        string.Join("\n\n", docs.Select((d, i) => $"[{i + 1}] (source: {d.Source})\n{d.Text}"));
}
