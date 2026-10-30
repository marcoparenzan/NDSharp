using System.Reflection;
using NDSharp.Frame;

return NdCli.Run(args, Console.Out, Console.Error);

/// <summary>The ndsharp command line. Kept in a class so the tests call it without spawning a process.</summary>
public static class NdCli
{
    public const string Usage = """
        ndsharp — NDSharp command line

          ndsharp info <file.csv>                 columns, types, non-null counts
          ndsharp head <file.csv> [-n N]          first N rows (default 5)
          ndsharp describe <file.csv>             count/mean/std/min/quartiles/max per column
          ndsharp convert <in.csv> <out.json>     write the table as JSON (records) or CSV, by output extension
          ndsharp selftest                        run a built-in check of the Frame library
          ndsharp --version | --help

        Options: --sep <c> field separator for CSV input (default ',').
        Exit codes: 0 ok, 1 unknown command, 2 usage or input error.
        """;

    public static int Run(string[] args, TextWriter output, TextWriter error)
    {
        if (args.Length == 0 || args[0] is "help" or "--help" or "-h") { output.WriteLine(Usage); return 0; }
        try
        {
            switch (args[0])
            {
                case "--version" or "-v" or "version":
                    output.WriteLine(typeof(NdCli).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion.Split('+')[0] ?? "unknown");
                    return 0;
                case "info": output.Write(Info.Describe(Load(args, 1))); return 0;
                case "head": output.WriteLine(Formatter.FrameRepr(Load(args, 1).Head(Count(args, 5)))); return 0;
                case "describe": return Describe(Load(args, 1), output);
                case "convert": return Convert(args, output);
                case "selftest": return SelfTest(output, error);
                default:
                    error.WriteLine($"Unknown command '{args[0]}'.");
                    error.WriteLine(Usage);
                    return 1;
            }
        }
        catch (Exception ex) when (ex is ArgumentException or IOException or FormatException or UnauthorizedAccessException)
        {
            error.WriteLine($"Error: {ex.Message}");
            return 2;
        }
    }

    private static string? Option(string[] args, string name)
    {
        var i = Array.IndexOf(args, name);
        if (i < 0) return null;
        return i + 1 < args.Length ? args[i + 1] : throw new ArgumentException($"{name} needs a value.");
    }

    private static int Count(string[] args, int fallback)
    {
        var text = Option(args, "-n");
        if (text is null) return fallback;
        return int.TryParse(text, out var n) && n >= 0 ? n : throw new ArgumentException($"-n must be a non-negative integer, got '{text}'.");
    }

    private static DataFrame Load(string[] args, int fileIndex)
    {
        if (args.Length <= fileIndex || args[fileIndex].StartsWith('-')) throw new ArgumentException("A CSV file is required.");
        var path = args[fileIndex];
        if (!File.Exists(path)) throw new IOException($"File not found: {path}");
        var sep = Option(args, "--sep");
        if (sep is { Length: not 1 }) throw new ArgumentException("--sep must be a single character.");
        return FromCsv(File.ReadAllText(path), sep is null ? ',' : sep[0]);
    }

    public static DataFrame FromCsv(string text, char sep = ',')
    {
        var (cols, names) = Csv.Read(text, new Csv.ReadOptions { Sep = sep });
        return new DataFrame(cols, new NDSharp.Frame.Index(Column.FromObjects(names.ToArray())));
    }

    private static int Describe(DataFrame df, TextWriter output)
    {
        for (var i = 0; i < df.NCols; i++)
        {
            var column = df.GetColumn(i);
            output.WriteLine(Formatter.SeriesRepr(FrameOps.Describe(column)));
        }
        return 0;
    }

    private static int Convert(string[] args, TextWriter output)
    {
        if (args.Length < 3) throw new ArgumentException("convert needs an input and an output file.");
        var df = Load(args, 1);
        var target = args[2];
        var ext = Path.GetExtension(target).ToLowerInvariant();
        string text = ext switch
        {
            ".json" => NDSharp.Frame.Json.ToJson(df, "records", 10, 2, false, false, v => v.ToString("R", System.Globalization.CultureInfo.InvariantCulture)),
            ".csv" => ToCsv(df),
            _ => throw new ArgumentException($"Unsupported output extension '{ext}'. Use .json or .csv."),
        };
        File.WriteAllText(target, text);
        output.WriteLine($"Wrote {df.NRows} rows x {df.NCols} columns to {target}");
        return 0;
    }

    private static string ToCsv(DataFrame df)
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine(string.Join(",", df.Columns.Items().Select(n => Csv.Quote(n?.ToString() ?? "", ',', '"'))));
        for (var r = 0; r < df.NRows; r++)
        {
            var cells = new string[df.NCols];
            for (var c = 0; c < df.NCols; c++)
                cells[c] = Csv.Quote(Cell(df.Data[c][r]), ',', '"');
            sb.AppendLine(string.Join(",", cells));
        }
        return sb.ToString();
    }

    private static string Cell(object? value) => value switch
    {
        null or NAValue => "",
        double d when double.IsNaN(d) => "",
        IFormattable f => f.ToString(null, System.Globalization.CultureInfo.InvariantCulture),
        _ => value.ToString() ?? "",
    };

    private static int SelfTest(TextWriter output, TextWriter error)
    {
        var df = FromCsv("a,b\n1,x\n2,y\n3,z\n");
        var ok = df.NRows == 3 && df.NCols == 2 && df.GetColumn(0).Length == 3;
        output.WriteLine(ok ? "selftest: ok" : "selftest: FAILED");
        if (!ok) error.WriteLine("Frame read a 3x2 CSV incorrectly.");
        return ok ? 0 : 1;
    }
}
