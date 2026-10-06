using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;

namespace StudentMcpServer.Storage;

public sealed record StudentFile(string StudentId, string FileName, long SizeBytes, DateTime LastModifiedUtc);

/// <summary>
/// The "file storage" scenario: each student has a folder of documents (transcripts, advisor
/// notes, assignment drafts). This reads from a local folder; the same shape works for cloud
/// storage like Azure Blob Storage, with one container folder per student.
/// </summary>
public sealed class StudentFileStore(string rootPath, ILogger<StudentFileStore>? logger = null)
{
    private readonly ILogger _logger = logger ?? NullLogger<StudentFileStore>.Instance;

    private static readonly HashSet<string> AllowedExtensions = [".md", ".txt", ".csv", ".json"];
    private const long MaxReadBytes = 256 * 1024;

    public string RootPath { get; } = Path.GetFullPath(rootPath);

    public IReadOnlyList<StudentFile> ListFiles(string studentId)
    {
        var folder = StudentFolder(studentId);
        if (!Directory.Exists(folder)) return [];
        return new DirectoryInfo(folder).EnumerateFiles()
            .Where(f => AllowedExtensions.Contains(f.Extension.ToLowerInvariant()))
            .OrderBy(f => f.Name, StringComparer.Ordinal)
            .Select(f => new StudentFile(NormalizeId(studentId), f.Name, f.Length, f.LastWriteTimeUtc))
            .ToList();
    }

    /// <summary>Reads one file. Throws <see cref="FileAccessException"/> for anything not allowed.</summary>
    public string ReadFile(string studentId, string fileName)
    {
        try
        {
            var text = ReadFileUnchecked(studentId, fileName);
            _logger.LogDebug("Read {FileName} for {StudentId} ({Length} chars)", fileName, studentId, text.Length);
            return text;
        }
        catch (FileAccessException ex)
        {
            // Refusals are worth seeing in the logs: they may be a confused model or a probing attempt.
            _logger.LogWarning("Refused file read for {StudentId}/{FileName}: {Reason}", studentId, fileName, ex.Message);
            throw;
        }
    }

    private string ReadFileUnchecked(string studentId, string fileName)
    {
        var folder = StudentFolder(studentId);
        var fullPath = Path.GetFullPath(Path.Combine(folder, fileName));

        // Security: the AI model chooses fileName, so make sure "../" tricks can't escape the student's folder.
        if (!fullPath.StartsWith(folder + Path.DirectorySeparatorChar, StringComparison.Ordinal))
            throw new FileAccessException($"'{fileName}' is outside {NormalizeId(studentId)}'s folder.");
        if (!AllowedExtensions.Contains(Path.GetExtension(fullPath).ToLowerInvariant()))
            throw new FileAccessException($"Only {string.Join(", ", AllowedExtensions)} files can be read.");
        if (!File.Exists(fullPath))
            throw new FileAccessException($"{NormalizeId(studentId)} has no file named '{fileName}'. Use list_student_files to see what exists.");
        if (new FileInfo(fullPath).Length > MaxReadBytes)
            throw new FileAccessException($"'{fileName}' is larger than {MaxReadBytes / 1024} KB.");
        return File.ReadAllText(fullPath);
    }

    private string StudentFolder(string studentId)
    {
        var id = NormalizeId(studentId);
        if (id.Length == 0 || !id.All(char.IsLetterOrDigit))
            throw new FileAccessException($"'{studentId}' is not a valid student id.");
        return Path.Combine(RootPath, id);
    }

    private static string NormalizeId(string studentId) => studentId.Trim().ToUpperInvariant();
}

public sealed class FileAccessException(string message) : Exception(message);
