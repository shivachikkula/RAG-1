using System.ComponentModel;
using ModelContextProtocol;
using ModelContextProtocol.Server;
using StudentMcpServer.Data;
using StudentMcpServer.ExternalApi;
using StudentMcpServer.Storage;

namespace StudentMcpServer.Tools;

/// <summary>Combined scenario: one call that gathers from the database, file storage and the external API.</summary>
[McpServerToolType]
public static class StudentOverviewTools
{
    [McpServerTool(Name = "get_student_overview", Title = "Student overview", ReadOnly = true, OpenWorld = true)]
    [Description("Everything about a student in one call: profile, transcript and GPA (database), their documents (file storage), and suggested textbooks for courses in progress (Open Library API).")]
    public static async Task<object> GetStudentOverview(
        StudentDatabase db,
        StudentFileStore files,
        TextbookApiClient api,
        [Description("Student id, e.g. S1001")] string studentId,
        CancellationToken cancellationToken = default)
    {
        var transcript = db.GetTranscript(studentId) ?? throw new McpException($"No student with id '{studentId}'.");

        var inProgress = transcript.Courses.Where(c => c.Grade is null).ToList();
        var reading = new List<object>();
        string? apiNote = null;
        foreach (var course in inProgress)
        {
            var topic = db.GetCourse(course.CourseCode)!.TextbookTopic;
            try
            {
                reading.Add(new { course = course.CourseCode, textbooks = await api.SearchAsync(topic, 3, cancellationToken) });
            }
            catch (Exception ex) when (ex is ExternalApiException or HttpRequestException or TaskCanceledException)
            {
                // The overview is still useful without book suggestions, so don't fail the whole call.
                apiNote = "Textbook suggestions are unavailable right now (Open Library could not be reached).";
                break;
            }
        }

        return new
        {
            profile = transcript.Student,
            gpa = transcript.Gpa,
            completedCredits = transcript.CompletedCredits,
            completedCourses = transcript.Courses.Where(c => c.Grade is not null),
            inProgressCourses = inProgress,
            documents = files.ListFiles(studentId).Select(f => f.FileName),
            suggestedReading = reading,
            note = apiNote,
        };
    }
}
