using System.ComponentModel;
using ModelContextProtocol.Server;

namespace StudentMcpServer.Prompts;

/// <summary>Prompts are ready-made instructions a user can pick from a menu in their AI app.</summary>
[McpServerPromptType]
public static class StudentPrompts
{
    [McpServerPrompt(Name = "student_progress_report", Title = "Student progress report")]
    [Description("Write an academic progress report for one student.")]
    public static string ProgressReport([Description("Student id, e.g. S1001")] string studentId) =>
        $"""
        Write a short academic progress report for student {studentId}.
        1. Call get_student_overview for their profile, GPA, courses and documents.
        2. Read their advisor notes with read_student_file if they have any.
        3. Report: current standing (GPA and credits), strengths, concerns, courses in progress,
           and two or three concrete next steps. Suggest textbooks for courses in progress if available.
        Keep it under 300 words, written for the student's academic advisor.
        """;

    [McpServerPrompt(Name = "study_plan", Title = "Study plan for a course")]
    [Description("Create a study plan for a student taking a specific course.")]
    public static string StudyPlan(
        [Description("Student id, e.g. S1003")] string studentId,
        [Description("Course code, e.g. MATH150")] string courseCode) =>
        $"""
        Create a 4-week study plan for student {studentId} in {courseCode}.
        Use get_student_transcript to see their background, list_courses for the course details,
        and recommend_textbooks_for_course for reading material. Tailor the plan to their past grades.
        """;
}
