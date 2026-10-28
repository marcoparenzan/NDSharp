using System.Reflection;
using System.Text.Json;
using Xunit;

public class NotebookTests
{
    [Theory]
    [InlineData(0)] [InlineData(1)] [InlineData(2)] [InlineData(3)]
    [InlineData(4)] [InlineData(5)] [InlineData(6)] [InlineData(7)]
    [InlineData(8)] [InlineData(9)] [InlineData(10)] [InlineData(11)]
    [InlineData(12)] [InlineData(13)]
    public void Example_executes(int number)
    {
        var method = typeof(Validation).GetMethod($"Sample{number}");
        Assert.NotNull(method);
        method.Invoke(null, null);
    }

    [Fact]
    public void Notebooks_have_csharp_kernel_and_cleared_outputs()
    {
        static string SourceDirectory([System.Runtime.CompilerServices.CallerFilePath] string source = "") => Path.GetDirectoryName(source)!;
        var root = new DirectoryInfo(SourceDirectory());
        while (root != null && !File.Exists(Path.Combine(root.FullName, "src", "NDSharp.slnx"))) root = root.Parent;
        Assert.NotNull(root);
        var files = Directory.GetFiles(Path.Combine(root.FullName, "src", "notebooks"), "*.ipynb");
        Assert.Equal(14, files.Length);
        foreach (var file in files)
        {
            using var json = JsonDocument.Parse(File.ReadAllText(file));
            var notebook = json.RootElement;
            Assert.Equal(4, notebook.GetProperty("nbformat").GetInt32());
            Assert.Equal(".net-csharp", notebook.GetProperty("metadata").GetProperty("kernelspec").GetProperty("name").GetString());
            foreach (var cell in notebook.GetProperty("cells").EnumerateArray())
                if (cell.GetProperty("cell_type").GetString() == "code")
                {
                    Assert.Equal(JsonValueKind.Null, cell.GetProperty("execution_count").ValueKind);
                    Assert.Equal(0, cell.GetProperty("outputs").GetArrayLength());
                    Assert.NotEmpty(cell.GetProperty("source").EnumerateArray());
                }
        }
    }
}
