using System.Text;
using JupyterSharp.Kernels.Abstractions;
using NDSharp.JupyterSharp.Kernel;

namespace NDSharp.JupyterSharp.Kernel.Tests;

public class KernelTests
{
    sealed class Sink : IKernelOutputSink
    {
        public StringBuilder Text { get; } = new();
        public List<string> Errors { get; } = [];
        public void WriteText(string text) => Text.Append(text);
        public void WriteHtml(string html) => Text.Append(html);
        public void WriteError(string message, string? stackTrace = null) => Errors.Add(message);
    }

    static async Task<Sink> Run(NDSharpKernel k, params string[] cells)
    {
        var s = new Sink();
        foreach (var c in cells) await k.ExecuteAsync(c, s, CancellationToken.None);
        return s;
    }

    [Fact]
    public void PluginAdvertisesTheKernelId()
    {
        Assert.Equal("ndsharp", new NDSharpKernelPlugin().KernelId);
        Assert.Equal("ndsharp", new NDSharpKernel().Id);
    }

    [Fact]
    public async Task NDSharpIsAvailableWithoutRefOrUsing()
    {
        var s = await Run(new NDSharpKernel(), "var a = np.Reshape(np.Arange(0, 6), 2, 3); np.Sum(a).ToString()");
        Assert.Empty(s.Errors);
        Assert.Contains("15", s.Text.ToString());
    }

    [Fact]
    public async Task TheBridgeToMathemaSharpIsImported()
    {
        var s = await Run(new NDSharpKernel(), "var v = NDArray.FromArray(new double[] { 1, 2, 3 }, 3).ToVector(); v.Length");
        Assert.Empty(s.Errors);
        Assert.Contains("3", s.Text.ToString());
    }

    [Fact]
    public async Task StatePersistsAcrossCells()
    {
        var s = await Run(new NDSharpKernel(), "var x = np.Arange(0, 4);", "np.Sum(x).ToString()");
        Assert.Empty(s.Errors);
        Assert.Contains("6", s.Text.ToString());
    }

    [Fact]
    public async Task CompilationErrorsAreReported()
    {
        var s = await Run(new NDSharpKernel(), "np.NoSuchFunction()");
        Assert.Single(s.Errors);
    }
}
