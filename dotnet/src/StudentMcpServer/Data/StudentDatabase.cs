using Microsoft.Data.Sqlite;

namespace StudentMcpServer.Data;

public sealed record Student(string Id, string Name, string Email, string Major, int Year, string AdvisorName);

public sealed record Course(string Code, string Title, string Department, int Credits, string Instructor, string TextbookTopic);

public sealed record CourseGrade(string CourseCode, string CourseTitle, int Credits, string Term, string? Grade);

public sealed record Transcript(Student Student, IReadOnlyList<CourseGrade> Courses, double? Gpa, int CompletedCredits);

public enum EnrollResult { Enrolled, AlreadyEnrolled, StudentNotFound, CourseNotFound }

/// <summary>
/// The "database" scenario: students, courses and grades in SQLite. SQLite keeps the whole
/// database in one file, so there's no database server to install. On first run the file is
/// created and filled with sample data.
/// </summary>
public sealed class StudentDatabase
{
    private readonly string _connectionString;

    public StudentDatabase(string databasePath)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(databasePath))!);
        _connectionString = new SqliteConnectionStringBuilder { DataSource = databasePath, Pooling = false }.ToString();
        EnsureCreated();
    }

    private SqliteConnection Open()
    {
        var connection = new SqliteConnection(_connectionString);
        connection.Open();
        return connection;
    }

    public Student? GetStudent(string studentId)
    {
        using var connection = Open();
        using var cmd = connection.CreateCommand();
        // Values are always passed as parameters ($id), never pasted into the SQL text,
        // so input from the AI model can't change the query (SQL injection).
        cmd.CommandText = "SELECT id, name, email, major, year, advisor FROM students WHERE id = $id";
        cmd.Parameters.AddWithValue("$id", studentId.Trim().ToUpperInvariant());
        using var reader = cmd.ExecuteReader();
        return reader.Read() ? ReadStudent(reader) : null;
    }

    public IReadOnlyList<Student> SearchStudents(string? name, string? major, int limit)
    {
        using var connection = Open();
        using var cmd = connection.CreateCommand();
        cmd.CommandText = """
            SELECT id, name, email, major, year, advisor FROM students
            WHERE ($name = '' OR name LIKE '%' || $name || '%')
              AND ($major = '' OR major LIKE '%' || $major || '%')
            ORDER BY name LIMIT $limit
            """;
        cmd.Parameters.AddWithValue("$name", name?.Trim() ?? "");
        cmd.Parameters.AddWithValue("$major", major?.Trim() ?? "");
        cmd.Parameters.AddWithValue("$limit", Math.Clamp(limit, 1, 50));
        using var reader = cmd.ExecuteReader();
        var students = new List<Student>();
        while (reader.Read()) students.Add(ReadStudent(reader));
        return students;
    }

    public IReadOnlyList<Course> ListCourses(string? department)
    {
        using var connection = Open();
        using var cmd = connection.CreateCommand();
        cmd.CommandText = """
            SELECT code, title, department, credits, instructor, textbook_topic FROM courses
            WHERE $dept = '' OR department LIKE '%' || $dept || '%' ORDER BY code
            """;
        cmd.Parameters.AddWithValue("$dept", department?.Trim() ?? "");
        using var reader = cmd.ExecuteReader();
        var courses = new List<Course>();
        while (reader.Read()) courses.Add(ReadCourse(reader));
        return courses;
    }

    public Course? GetCourse(string courseCode) =>
        ListCourses(null).FirstOrDefault(c => c.Code.Equals(courseCode.Trim(), StringComparison.OrdinalIgnoreCase));

    public Transcript? GetTranscript(string studentId)
    {
        var student = GetStudent(studentId);
        if (student is null) return null;

        using var connection = Open();
        using var cmd = connection.CreateCommand();
        cmd.CommandText = """
            SELECT c.code, c.title, c.credits, e.term, e.grade
            FROM enrollments e JOIN courses c ON c.code = e.course_code
            WHERE e.student_id = $id ORDER BY e.term, c.code
            """;
        cmd.Parameters.AddWithValue("$id", student.Id);
        using var reader = cmd.ExecuteReader();
        var courses = new List<CourseGrade>();
        while (reader.Read())
            courses.Add(new CourseGrade(reader.GetString(0), reader.GetString(1), reader.GetInt32(2), reader.GetString(3), reader.IsDBNull(4) ? null : reader.GetString(4)));

        // GPA on a 4.0 scale, weighted by credits; courses still in progress (no grade) don't count.
        var graded = courses.Where(c => c.Grade is not null && GradePoints.ContainsKey(c.Grade)).ToList();
        var credits = graded.Sum(c => c.Credits);
        double? gpa = credits == 0 ? null : Math.Round(graded.Sum(c => GradePoints[c.Grade!] * c.Credits) / credits, 2);
        return new Transcript(student, courses, gpa, credits);
    }

    public EnrollResult Enroll(string studentId, string courseCode, string term)
    {
        if (GetStudent(studentId) is not { } student) return EnrollResult.StudentNotFound;
        if (GetCourse(courseCode) is not { } course) return EnrollResult.CourseNotFound;

        using var connection = Open();
        using var cmd = connection.CreateCommand();
        cmd.CommandText = "INSERT OR IGNORE INTO enrollments (student_id, course_code, term, grade) VALUES ($s, $c, $t, NULL)";
        cmd.Parameters.AddWithValue("$s", student.Id);
        cmd.Parameters.AddWithValue("$c", course.Code);
        cmd.Parameters.AddWithValue("$t", term.Trim());
        return cmd.ExecuteNonQuery() == 1 ? EnrollResult.Enrolled : EnrollResult.AlreadyEnrolled;
    }

    private static readonly Dictionary<string, double> GradePoints = new()
    {
        ["A"] = 4.0, ["A-"] = 3.7, ["B+"] = 3.3, ["B"] = 3.0, ["B-"] = 2.7,
        ["C+"] = 2.3, ["C"] = 2.0, ["C-"] = 1.7, ["D"] = 1.0, ["F"] = 0.0,
    };

    private static Student ReadStudent(SqliteDataReader r) =>
        new(r.GetString(0), r.GetString(1), r.GetString(2), r.GetString(3), r.GetInt32(4), r.GetString(5));

    private static Course ReadCourse(SqliteDataReader r) =>
        new(r.GetString(0), r.GetString(1), r.GetString(2), r.GetInt32(3), r.GetString(4), r.GetString(5));

    private void EnsureCreated()
    {
        using var connection = Open();
        using var cmd = connection.CreateCommand();
        cmd.CommandText = """
            CREATE TABLE IF NOT EXISTS students (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, email TEXT NOT NULL,
                major TEXT NOT NULL, year INTEGER NOT NULL, advisor TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS courses (
                code TEXT PRIMARY KEY, title TEXT NOT NULL, department TEXT NOT NULL,
                credits INTEGER NOT NULL, instructor TEXT NOT NULL, textbook_topic TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS enrollments (
                student_id TEXT NOT NULL REFERENCES students(id),
                course_code TEXT NOT NULL REFERENCES courses(code),
                term TEXT NOT NULL, grade TEXT,
                PRIMARY KEY (student_id, course_code, term));
            """;
        cmd.ExecuteNonQuery();

        cmd.CommandText = "SELECT COUNT(*) FROM students";
        if (Convert.ToInt32(cmd.ExecuteScalar()) == 0)
            Seed(connection);
    }

    private static void Seed(SqliteConnection connection)
    {
        using var tx = connection.BeginTransaction();
        void Exec(string sql, params object[] values)
        {
            using var cmd = connection.CreateCommand();
            cmd.Transaction = tx;
            cmd.CommandText = sql;
            for (var i = 0; i < values.Length; i++) cmd.Parameters.AddWithValue($"$p{i}", values[i]);
            cmd.ExecuteNonQuery();
        }

        foreach (var s in new[]
        {
            new Student("S1001", "Aisha Khan", "aisha.khan@example.edu", "Computer Science", 2, "Dr. Elena Rossi"),
            new Student("S1002", "Diego Martinez", "diego.martinez@example.edu", "Biology", 3, "Dr. Samuel Okafor"),
            new Student("S1003", "Mei Chen", "mei.chen@example.edu", "Mathematics", 1, "Dr. Elena Rossi"),
            new Student("S1004", "Liam O'Brien", "liam.obrien@example.edu", "History", 4, "Dr. Hannah Weiss"),
            new Student("S1005", "Priya Patel", "priya.patel@example.edu", "Computer Science", 3, "Dr. Samuel Okafor"),
        })
            Exec("INSERT INTO students VALUES ($p0, $p1, $p2, $p3, $p4, $p5)", s.Id, s.Name, s.Email, s.Major, s.Year, s.AdvisorName);

        foreach (var c in new[]
        {
            new Course("CS101", "Introduction to Programming", "Computer Science", 4, "Prof. Alan Brooks", "python programming"),
            new Course("CS201", "Data Structures and Algorithms", "Computer Science", 4, "Prof. Alan Brooks", "data structures algorithms"),
            new Course("MATH150", "Calculus I", "Mathematics", 4, "Prof. Grace Liu", "calculus"),
            new Course("STAT200", "Introduction to Statistics", "Mathematics", 3, "Prof. Grace Liu", "introductory statistics"),
            new Course("BIO110", "Cell Biology", "Biology", 4, "Prof. Maria Santos", "cell biology"),
            new Course("HIST210", "Modern European History", "History", 3, "Prof. James Carter", "modern european history"),
        })
            Exec("INSERT INTO courses VALUES ($p0, $p1, $p2, $p3, $p4, $p5)", c.Code, c.Title, c.Department, c.Credits, c.Instructor, c.TextbookTopic);

        foreach (var (student, course, term, grade) in new (string, string, string, string?)[]
        {
            ("S1001", "CS101", "2025-Fall", "A"), ("S1001", "MATH150", "2025-Fall", "B+"),
            ("S1001", "CS201", "2026-Spring", "A-"), ("S1001", "STAT200", "2026-Fall", null),
            ("S1002", "BIO110", "2024-Fall", "A"), ("S1002", "STAT200", "2025-Spring", "B"),
            ("S1002", "MATH150", "2025-Fall", "C+"),
            ("S1003", "MATH150", "2026-Fall", null), ("S1003", "CS101", "2026-Fall", null),
            ("S1004", "HIST210", "2024-Spring", "A"), ("S1004", "STAT200", "2025-Fall", "B-"),
            ("S1005", "CS101", "2024-Fall", "B"), ("S1005", "CS201", "2025-Spring", "B+"),
            ("S1005", "MATH150", "2025-Fall", "A-"),
        })
            Exec("INSERT INTO enrollments VALUES ($p0, $p1, $p2, $p3)", student, course, term, (object?)grade ?? DBNull.Value);

        tx.Commit();
    }
}
