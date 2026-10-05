using System.Net.Http.Json;
using System.Text.Json.Serialization;

namespace StudentMcpServer.ExternalApi;

public sealed record Textbook(string Title, IReadOnlyList<string> Authors, int? FirstPublished, string? Isbn, string? Url);

/// <summary>
/// The "external API" scenario: finds textbooks for a course topic using Open Library's free,
/// keyless search API (https://openlibrary.org/developers/api). The HttpClient is created by
/// the framework (AddHttpClient), with a base address and timeout set in StudentServerSetup.
/// </summary>
public sealed class TextbookApiClient(HttpClient http)
{
    public async Task<IReadOnlyList<Textbook>> SearchAsync(string topic, int limit, CancellationToken ct)
    {
        limit = Math.Clamp(limit, 1, 10);
        var url = $"search.json?q={Uri.EscapeDataString(topic)}&limit={limit}&fields=key,title,author_name,first_publish_year,isbn";
        using var response = await http.GetAsync(url, ct);
        if (!response.IsSuccessStatusCode)
            throw new ExternalApiException($"Open Library returned HTTP {(int)response.StatusCode}.");

        var body = await response.Content.ReadFromJsonAsync<SearchResponse>(ct)
                   ?? throw new ExternalApiException("Open Library returned an empty response.");
        return (body.Docs ?? [])
            .Where(d => !string.IsNullOrWhiteSpace(d.Title))
            .Select(d => new Textbook(
                d.Title!,
                d.AuthorName ?? [],
                d.FirstPublishYear,
                d.Isbn?.FirstOrDefault(),
                d.Key is null ? null : $"https://openlibrary.org{d.Key}"))
            .ToList();
    }

    private sealed record SearchResponse([property: JsonPropertyName("docs")] List<SearchDoc>? Docs);

    private sealed record SearchDoc(
        [property: JsonPropertyName("key")] string? Key,
        [property: JsonPropertyName("title")] string? Title,
        [property: JsonPropertyName("author_name")] List<string>? AuthorName,
        [property: JsonPropertyName("first_publish_year")] int? FirstPublishYear,
        [property: JsonPropertyName("isbn")] List<string>? Isbn);
}

public sealed class ExternalApiException(string message) : Exception(message);
