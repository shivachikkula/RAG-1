using System.ComponentModel;
using ModelContextProtocol;
using ModelContextProtocol.Server;
using StudentMcpServer.Data;
using StudentMcpServer.ExternalApi;

namespace StudentMcpServer.Tools;

/// <summary>External API scenario: find textbooks through the Open Library search API.</summary>
[McpServerToolType]
public static class TextbookTools
{
    [McpServerTool(Name = "find_textbooks", Title = "Find textbooks", ReadOnly = true, OpenWorld = true)]
    [Description("Search the Open Library catalog (external internet API) for books on a topic. Returns title, authors, first publication year, ISBN and a link.")]
    public static async Task<IReadOnlyList<Textbook>> FindTextbooks(
        TextbookApiClient api,
        [Description("Topic or title to search for, e.g. 'linear algebra'")] string topic,
        [Description("Maximum results (1-10).")] int limit = 5,
        CancellationToken cancellationToken = default) =>
        await CallApiAsync(() => api.SearchAsync(topic, limit, cancellationToken));

    [McpServerTool(Name = "recommend_textbooks_for_course", Title = "Recommend course textbooks", ReadOnly = true, OpenWorld = true)]
    [Description("Recommend textbooks for a course: looks up the course's topic in the database, then searches Open Library for it.")]
    public static async Task<object> RecommendTextbooksForCourse(
        StudentDatabase db,
        TextbookApiClient api,
        [Description("Course code, e.g. MATH150")] string courseCode,
        CancellationToken cancellationToken = default)
    {
        var course = db.GetCourse(courseCode) ?? throw new McpException($"No course with code '{courseCode}'. Use list_courses to see the catalog.");
        var books = await CallApiAsync(() => api.SearchAsync(course.TextbookTopic, 5, cancellationToken));
        return new { course = course.Code, course.Title, searchedFor = course.TextbookTopic, textbooks = books };
    }

    /// <summary>Turns network failures into a clear message for the AI instead of a generic error.</summary>
    internal static async Task<T> CallApiAsync<T>(Func<Task<T>> call)
    {
        try { return await call(); }
        catch (ExternalApiException ex) { throw new McpException(ex.Message); }
        catch (HttpRequestException ex) { throw new McpException($"Could not reach Open Library: {ex.Message}"); }
        catch (TaskCanceledException) { throw new McpException("Open Library did not respond in time. Try again later."); }
    }
}
