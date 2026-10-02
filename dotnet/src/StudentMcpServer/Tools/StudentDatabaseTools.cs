using System.ComponentModel;
using ModelContextProtocol;
using ModelContextProtocol.Server;
using StudentMcpServer.Data;

namespace StudentMcpServer.Tools;

/// <summary>Database scenario: look up students, courses and grades; enroll a student.</summary>
[McpServerToolType]
public static class StudentDatabaseTools
{
    [McpServerTool(Name = "search_students", Title = "Search students", ReadOnly = true)]
    [Description("Search students by (part of) their name and/or major. Leave a filter empty to ignore it. Returns id, name, email, major, year and advisor.")]
    public static IReadOnlyList<Student> SearchStudents(
        StudentDatabase db,
        [Description("Part of the student's name, e.g. 'Khan'. Optional.")] string? name = null,
        [Description("Part of the major, e.g. 'Computer'. Optional.")] string? major = null,
        [Description("Maximum results (1-50).")] int limit = 10) =>
        db.SearchStudents(name, major, limit);

    [McpServerTool(Name = "get_student", Title = "Get student", ReadOnly = true)]
    [Description("Get one student's profile by student id (e.g. S1001).")]
    public static Student GetStudent(StudentDatabase db, [Description("Student id, e.g. S1001")] string studentId) =>
        db.GetStudent(studentId) ?? throw new McpException($"No student with id '{studentId}'. Use search_students to find the id.");

    [McpServerTool(Name = "get_student_transcript", Title = "Get transcript", ReadOnly = true)]
    [Description("Get a student's courses with term and grade, their credit-weighted GPA (4.0 scale) and completed credits. Courses with no grade are in progress.")]
    public static Transcript GetStudentTranscript(StudentDatabase db, [Description("Student id, e.g. S1001")] string studentId) =>
        db.GetTranscript(studentId) ?? throw new McpException($"No student with id '{studentId}'.");

    [McpServerTool(Name = "list_courses", Title = "List courses", ReadOnly = true)]
    [Description("List the course catalog: code, title, department, credits, instructor. Optionally filter by department.")]
    public static IReadOnlyList<Course> ListCourses(
        StudentDatabase db,
        [Description("Part of a department name, e.g. 'Math'. Optional.")] string? department = null) =>
        db.ListCourses(department);

    [McpServerTool(Name = "enroll_student", Title = "Enroll student in a course", Destructive = false, Idempotent = true)]
    [Description("Enroll a student in a course for a term. This CHANGES the database. Enrolling twice in the same course and term does nothing.")]
    public static string EnrollStudent(
        StudentDatabase db,
        [Description("Student id, e.g. S1003")] string studentId,
        [Description("Course code, e.g. STAT200")] string courseCode,
        [Description("Term, e.g. 2027-Spring")] string term) =>
        db.Enroll(studentId, courseCode, term) switch
        {
            EnrollResult.Enrolled => $"Enrolled {studentId.ToUpperInvariant()} in {courseCode.ToUpperInvariant()} for {term}.",
            EnrollResult.AlreadyEnrolled => $"{studentId.ToUpperInvariant()} is already enrolled in {courseCode.ToUpperInvariant()} for {term}.",
            EnrollResult.StudentNotFound => throw new McpException($"No student with id '{studentId}'."),
            _ => throw new McpException($"No course with code '{courseCode}'. Use list_courses to see the catalog."),
        };
}
