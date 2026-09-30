using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using Microsoft.Extensions.AI;

namespace RagCore;

public static partial class TextUtil
{
    private static readonly HashSet<string> StopWords =
    [
        .. "a an and are as at be by for from has have how in is it its of on or that the this to was were what when where which who why will with does do did can".Split(' '),
    ];

    [GeneratedRegex("[a-z0-9]+")]
    private static partial Regex Word();

    public static List<string> Tokenize(string text) =>
        Word().Matches(text.ToLowerInvariant()).Select(m => m.Value).Where(t => !StopWords.Contains(t)).ToList();
}

/// <summary>
/// Offline, dependency-free embeddings: a hashed bag of words and word pairs.
/// Claude has no embeddings endpoint, so embeddings are pluggable. This implements the standard
/// .NET <see cref="IEmbeddingGenerator{TInput, TEmbedding}"/> interface, so you can swap in any
/// real embedding model (Azure OpenAI, Ollama, ONNX, ...) without changing the pipelines.
/// </summary>
public sealed class HashingEmbeddingGenerator(int dimensions = 1024) : IEmbeddingGenerator<string, Embedding<float>>
{
    public int Dimensions => dimensions;

    public Task<GeneratedEmbeddings<Embedding<float>>> GenerateAsync(
        IEnumerable<string> values, EmbeddingGenerationOptions? options = null, CancellationToken cancellationToken = default) =>
        Task.FromResult(new GeneratedEmbeddings<Embedding<float>>(values.Select(v => new Embedding<float>(Embed(v)))));

    private float[] Embed(string text)
    {
        var vector = new float[dimensions];
        var tokens = TextUtil.Tokenize(text);
        var features = tokens.Concat(tokens.Zip(tokens.Skip(1), (a, b) => $"{a}_{b}"));
        foreach (var feature in features)
        {
            var hash = BitConverter.ToUInt64(MD5.HashData(Encoding.UTF8.GetBytes(feature)), 0);
            vector[(int)(hash % (ulong)dimensions)] += (hash >> 63) == 1 ? 1f : -1f;
        }
        var norm = MathF.Sqrt(vector.Sum(x => x * x));
        if (norm > 0)
            for (var i = 0; i < vector.Length; i++) vector[i] /= norm;
        return vector;
    }

    public object? GetService(Type serviceType, object? serviceKey = null) => serviceType.IsInstanceOfType(this) ? this : null;

    public void Dispose() { }
}
