using System.Net;
using System.Text.Json;
using ModelContextProtocol.Protocol;

namespace StudentMcpServer.Tests;

public class StudentMcpServerTests
{
    private static JsonElement Json(CallToolResult result)
    {
        Assert.NotEqual(true, result.IsError);
        return JsonDocument.Parse(TestServer.TextOf(result)).RootElement;
    }

    // ---- discovery --------------------------------------------------------------------

    [Fact]
    public async Task ListsAllToolsWithSafetyAnnotations()
    {
        await using var server = await TestServer.StartAsync();
        var tools = (await server.Client.ListToolsAsync()).ToDictionary(t => t.Name);

        Assert.Equal(
            ["enroll_student", "find_textbooks", "get_student", "get_student_overview", "get_student_transcript",
             "list_courses", "list_student_files", "read_student_file", "recommend_textbooks_for_course", "search_students"],
            tools.Keys.Order());
        Assert.True(tools["get_student"].ProtocolTool.Annotations?.ReadOnlyHint);
        Assert.NotEqual(true, tools["enroll_student"].ProtocolTool.Annotations?.ReadOnlyHint);
        Assert.True(tools["find_textbooks"].ProtocolTool.Annotations?.OpenWorldHint);
        Assert.Contains("studentId", tools["get_student"].JsonSchema.GetRawText());
    }

    // ---- database ---------------------------------------------------------------------

    [Fact]
    public async Task GetsStudentFromDatabase()
    {
        await using var server = await TestServer.StartAsync();
        var student = Json(await server.CallAsync("get_student", ("studentId", "s1001")));
        Assert.Equal("Aisha Khan", student.GetProperty("name").GetString());
        Assert.Equal("Computer Science", student.GetProperty("major").GetString());
    }

    [Fact]
    public async Task UnknownStudentIsAClearError()
    {
        await using var server = await TestServer.StartAsync();
        var result = await server.CallAsync("get_student", ("studentId", "S9999"));
        Assert.True(result.IsError);
        Assert.Contains("No student with id 'S9999'", TestServer.TextOf(result));
    }

    [Fact]
    public async Task SearchesStudentsByMajor()
    {
        await using var server = await TestServer.StartAsync();
        var ids = Json(await server.CallAsync("search_students", ("major", "Computer"))).EnumerateArray().Select(s => s.GetProperty("id").GetString());
        Assert.Equal(["S1001", "S1005"], ids.Order());
    }

    [Fact]
    public async Task SearchTreatsSqlAsPlainText()
    {
        await using var server = await TestServer.StartAsync();
        var result = Json(await server.CallAsync("search_students", ("name", "' OR '1'='1")));
        Assert.Equal(0, result.GetArrayLength());
    }

    [Fact]
    public async Task TranscriptHasCreditWeightedGpa()
    {
        await using var server = await TestServer.StartAsync();
        var transcript = Json(await server.CallAsync("get_student_transcript", ("studentId", "S1001")));
        // A (4.0), B+ (3.3), A- (3.7), each 4 credits: 44 / 12 = 3.67. STAT200 is in progress.
        Assert.Equal(3.67, transcript.GetProperty("gpa").GetDouble());
        Assert.Equal(12, transcript.GetProperty("completedCredits").GetInt32());
        Assert.Equal(4, transcript.GetProperty("courses").GetArrayLength());
    }

    [Fact]
    public async Task EnrollsOnceAndIsIdempotent()
    {
        await using var server = await TestServer.StartAsync();
        var args = new[] { ("studentId", (object?)"S1003"), ("courseCode", "STAT200"), ("term", "2027-Spring") };

        Assert.StartsWith("Enrolled S1003 in STAT200", TestServer.TextOf(await server.CallAsync("enroll_student", args)));
        Assert.Contains("already enrolled", TestServer.TextOf(await server.CallAsync("enroll_student", args)));
        var courses = Json(await server.CallAsync("get_student_transcript", ("studentId", "S1003"))).GetProperty("courses");
        Assert.Contains(courses.EnumerateArray(), c => c.GetProperty("courseCode").GetString() == "STAT200");
    }

    [Fact]
    public async Task EnrollRejectsUnknownCourse()
    {
        await using var server = await TestServer.StartAsync();
        var result = await server.CallAsync("enroll_student", ("studentId", "S1003"), ("courseCode", "ART999"), ("term", "2027-Spring"));
        Assert.True(result.IsError);
        Assert.Contains("No course with code 'ART999'", TestServer.TextOf(result));
    }

    // ---- file storage -----------------------------------------------------------------

    [Fact]
    public async Task ListsAndReadsStudentFiles()
    {
        await using var server = await TestServer.StartAsync();
        var files = Json(await server.CallAsync("list_student_files", ("studentId", "S1001"))).EnumerateArray().Select(f => f.GetProperty("fileName").GetString());
        Assert.Equal(["advisor-notes.md", "research-statement-draft.txt", "schedule-2026-fall.csv"], files);

        var notes = TestServer.TextOf(await server.CallAsync("read_student_file", ("studentId", "S1001"), ("fileName", "advisor-notes.md")));
        Assert.Contains("Data Science Lab", notes);
    }

    [Theory]
    [InlineData("S1001", "../S1002/advisor-notes.md", "outside")]
    [InlineData("S1001", "/etc/passwd", "outside")]
    [InlineData("..", "S1002/advisor-notes.md", "not a valid student id")]
    [InlineData("S1001", "notes.exe", "Only")]
    [InlineData("S1001", "missing.md", "has no file named")]
    public async Task FileAccessIsRestrictedToTheStudentsFolder(string studentId, string fileName, string expectedError)
    {
        await using var server = await TestServer.StartAsync();
        var result = await server.CallAsync("read_student_file", ("studentId", studentId), ("fileName", fileName));
        Assert.True(result.IsError);
        Assert.Contains(expectedError, TestServer.TextOf(result));
    }

    // ---- external API -----------------------------------------------------------------

    [Fact]
    public async Task FindsTextbooksThroughTheExternalApi()
    {
        await using var server = await TestServer.StartAsync(FakeApi.Returning(TestServer.OpenLibraryJson(("Linear Algebra Done Right", "Sheldon Axler"))));
        var books = Json(await server.CallAsync("find_textbooks", ("topic", "linear algebra"), ("limit", 3)));

        Assert.Equal("Linear Algebra Done Right", books[0].GetProperty("title").GetString());
        Assert.Equal("https://openlibrary.org/works/OL0W", books[0].GetProperty("url").GetString());
        var request = Assert.Single(server.Api.Requests);
        Assert.Equal("https://openlibrary.org/search.json?q=linear%20algebra&limit=3&fields=key,title,author_name,first_publish_year,isbn", request.RequestUri!.AbsoluteUri);
        Assert.Contains("StudentMcpServer", request.Headers.UserAgent.ToString());
    }

    [Fact]
    public async Task RecommendsTextbooksUsingTheCourseTopicFromTheDatabase()
    {
        await using var server = await TestServer.StartAsync();
        var result = Json(await server.CallAsync("recommend_textbooks_for_course", ("courseCode", "MATH150")));
        Assert.Equal("calculus", result.GetProperty("searchedFor").GetString());
        Assert.Contains("q=calculus", server.Api.Requests[0].RequestUri!.Query);
    }

    [Fact]
    public async Task ApiFailureIsAClearError()
    {
        await using var server = await TestServer.StartAsync(FakeApi.Failing(HttpStatusCode.ServiceUnavailable));
        var result = await server.CallAsync("find_textbooks", ("topic", "calculus"));
        Assert.True(result.IsError);
        Assert.Contains("HTTP 503", TestServer.TextOf(result));
    }

    // ---- combined -----------------------------------------------------------------------

    [Fact]
    public async Task OverviewCombinesDatabaseFilesAndApi()
    {
        await using var server = await TestServer.StartAsync(FakeApi.Returning(TestServer.OpenLibraryJson(("Statistics", "David Freedman"))));
        var overview = Json(await server.CallAsync("get_student_overview", ("studentId", "S1001")));

        Assert.Equal("Aisha Khan", overview.GetProperty("profile").GetProperty("name").GetString());
        Assert.Equal(3.67, overview.GetProperty("gpa").GetDouble());
        Assert.Equal(3, overview.GetProperty("documents").GetArrayLength());
        var reading = overview.GetProperty("suggestedReading")[0];
        Assert.Equal("STAT200", reading.GetProperty("course").GetString());
        Assert.Equal("Statistics", reading.GetProperty("textbooks")[0].GetProperty("title").GetString());
    }

    [Fact]
    public async Task OverviewStillWorksWhenTheApiIsDown()
    {
        await using var server = await TestServer.StartAsync(FakeApi.Failing(HttpStatusCode.InternalServerError));
        var overview = Json(await server.CallAsync("get_student_overview", ("studentId", "S1001")));
        Assert.Equal(3.67, overview.GetProperty("gpa").GetDouble());
        Assert.Contains("unavailable", overview.GetProperty("note").GetString());
    }

    // ---- resources and prompts ----------------------------------------------------------

    [Fact]
    public async Task ReadsTranscriptAndFileResources()
    {
        await using var server = await TestServer.StartAsync();
        var transcript = await server.Client.ReadResourceAsync("student://S1002/transcript");
        Assert.Contains("Diego Martinez", Assert.IsType<TextResourceContents>(transcript.Contents[0]).Text);

        var file = await server.Client.ReadResourceAsync("student-files://S1003/welcome-letter.md");
        Assert.Contains("Mathematics program", Assert.IsType<TextResourceContents>(file.Contents[0]).Text);
    }

    [Fact]
    public async Task ProvidesPrompts()
    {
        await using var server = await TestServer.StartAsync();
        var prompts = (await server.Client.ListPromptsAsync()).Select(p => p.Name).Order();
        Assert.Equal(["student_progress_report", "study_plan"], prompts);

        var prompt = await server.Client.GetPromptAsync("student_progress_report", new Dictionary<string, object?> { ["studentId"] = "S1001" });
        Assert.Contains("student S1001", Assert.IsType<TextContentBlock>(prompt.Messages[0].Content).Text);
    }
}
