namespace NDSharp.Cli.Tests;

public class CliTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("ndsharp-cli-").FullName;
    private string Csv(string text) { var p = Path.Combine(_dir, Guid.NewGuid().ToString("N") + ".csv"); File.WriteAllText(p, text); return p; }
    public void Dispose() => Directory.Delete(_dir, true);

    private static (int Code, string Out, string Err) Run(params string[] args)
    {
        var o = new StringWriter(); var e = new StringWriter();
        return (NdCli.Run(args, o, e), o.ToString(), e.ToString());
    }

    [Fact]
    public void NoArgumentsPrintsUsageAndSucceeds()
    {
        var (code, output, _) = Run();
        Assert.Equal(0, code);
        Assert.Contains("ndsharp info", output);
    }

    [Fact]
    public void UnknownCommandExitsOne()
    {
        var (code, _, err) = Run("frobnicate");
        Assert.Equal(1, code);
        Assert.Contains("Unknown command", err);
    }

    [Fact]
    public void InfoListsColumnsAndCounts()
    {
        var (code, output, _) = Run("info", Csv("a,b\n1,x\n2,\n"));
        Assert.Equal(0, code);
        Assert.Contains("total 2 columns", output);
        Assert.Contains("1 non-null", output); // column b has one missing cell
    }

    [Fact]
    public void HeadHonoursN()
    {
        var (code, output, _) = Run("head", Csv("a\n1\n2\n3\n"), "-n", "2");
        Assert.Equal(0, code);
        Assert.Contains("2", output);
        Assert.DoesNotContain("\n2    3", output);
        Assert.Equal(3, output.Split('\n', StringSplitOptions.RemoveEmptyEntries).Length); // header + 2 rows
    }

    [Fact]
    public void DescribeReportsMeanOfANumericColumn()
    {
        var (code, output, _) = Run("describe", Csv("x\n1\n2\n3\n"));
        Assert.Equal(0, code);
        Assert.Contains("mean", output);
        Assert.Matches(@"mean\s+2\.0", output);
    }

    [Fact]
    public void ConvertWritesJsonAndCsvAndKeepsQuotedCommas()
    {
        var input = Csv("name,n\n\"Cy, Jr\",1\nAnn,\n");
        var json = Path.Combine(_dir, "o.json"); var csv = Path.Combine(_dir, "o.csv");
        Assert.Equal(0, Run("convert", input, json).Code);
        Assert.Equal(0, Run("convert", input, csv).Code);
        Assert.Contains("\"Cy, Jr\"", File.ReadAllText(json));
        var lines = File.ReadAllLines(csv);
        Assert.Equal("name,n", lines[0]);
        Assert.Equal("\"Cy, Jr\",1", lines[1]);
        Assert.Equal("Ann,", lines[2]); // missing value is empty, not NaN
    }

    [Theory]
    [InlineData("convert", "in.csv", "out.xlsx")]
    public void UnsupportedOutputExtensionExitsTwo(params string[] args)
    {
        args[1] = Csv("a\n1\n");
        var (code, _, err) = Run(args);
        Assert.Equal(2, code);
        Assert.Contains("Unsupported output extension", err);
    }

    [Fact]
    public void MissingFileAndBadOptionExitTwo()
    {
        Assert.Equal(2, Run("info", Path.Combine(_dir, "nope.csv")).Code);
        Assert.Equal(2, Run("head", Csv("a\n1\n"), "-n", "minus").Code);
        Assert.Equal(2, Run("info").Code);
    }

    [Fact]
    public void SelfTestPasses() => Assert.Equal(0, Run("selftest").Code);
}
