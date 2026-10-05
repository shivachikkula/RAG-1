using System.ComponentModel;
using ModelContextProtocol;
using ModelContextProtocol.Server;
using StudentMcpServer.Storage;

namespace StudentMcpServer.Tools;

/// <summary>File storage scenario: list and read a student's documents.</summary>
[McpServerToolType]
public static class StudentFileTools
{
    [McpServerTool(Name = "list_student_files", Title = "List student files", ReadOnly = true)]
    [Description("List the documents stored for a student (transcripts, advisor notes, assignment drafts) with size and last-modified time.")]
    public static IReadOnlyList<StudentFile> ListStudentFiles(StudentFileStore files, [Description("Student id, e.g. S1001")] string studentId)
    {
        try { return files.ListFiles(studentId); }
        catch (FileAccessException ex) { throw new McpException(ex.Message); }
    }

    [McpServerTool(Name = "read_student_file", Title = "Read student file", ReadOnly = true)]
    [Description("Read the text of one of a student's documents. Get the file name from list_student_files first.")]
    public static string ReadStudentFile(
        StudentFileStore files,
        [Description("Student id, e.g. S1001")] string studentId,
        [Description("File name exactly as returned by list_student_files, e.g. advisor-notes.md")] string fileName)
    {
        try { return files.ReadFile(studentId, fileName); }
        catch (FileAccessException ex) { throw new McpException(ex.Message); }
    }
}
