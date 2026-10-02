using System.ComponentModel;
using System.Text.Json;
using ModelContextProtocol;
using ModelContextProtocol.Server;
using StudentMcpServer.Data;
using StudentMcpServer.Storage;

namespace StudentMcpServer.Resources;

/// <summary>
/// Resources are read-only data an app can attach to a conversation directly (like attaching a
/// file), as opposed to tools, which the AI decides to call.
/// </summary>
[McpServerResourceType]
public static class StudentResources
{
    [McpServerResource(UriTemplate = "student://{studentId}/transcript", Name = "Student transcript", MimeType = "application/json")]
    [Description("A student's profile, courses, grades and GPA.")]
    public static string Transcript(StudentDatabase db, string studentId) =>
        JsonSerializer.Serialize(db.GetTranscript(studentId) ?? throw new McpException($"No student with id '{studentId}'."),
            new JsonSerializerOptions(JsonSerializerDefaults.Web) { WriteIndented = true });

    [McpServerResource(UriTemplate = "student-files://{studentId}/{fileName}", Name = "Student document", MimeType = "text/plain")]
    [Description("One document from a student's folder in file storage.")]
    public static string Document(StudentFileStore files, string studentId, string fileName)
    {
        try { return files.ReadFile(studentId, fileName); }
        catch (FileAccessException ex) { throw new McpException(ex.Message); }
    }
}
