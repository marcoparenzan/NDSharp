using System.Reflection;
using JupyterSharp.Kernels.Abstractions;
using JupyterSharp.Kernels.CSharp;

namespace NDSharp.JupyterSharp.Kernel;

/// <summary>
/// C# cells with NDSharp ready to use. It is the host's C# kernel (Roslyn scripting, state shared across cells) started with the NDSharp,
/// NDSharp.Mathema, NDSharp.Frame, NDSharp.Plot, NDSharp.Image, NDSharp.Wavelets and MathemaSharp assemblies referenced, and
/// <c>using NDSharp; using NDSharp.Mathema; using System.Numerics;</c> already in effect. The other namespaces are referenced but not
/// imported (their names can collide): a cell adds the <c>using</c> it wants.
/// </summary>
public sealed class NDSharpKernel : IKernel, IVariableInjectable
{
    public string Id => "ndsharp";

    private readonly CSharpKernel _inner = new(null, LoadLibraries(), ["NDSharp", "NDSharp.Mathema", "System.Numerics"]);

    /// <summary>Every NDSharp.* and MathemaSharp.* assembly in the plugin's folder (the ones the cells can use), loaded into the process once.</summary>
    private static List<Assembly> LoadLibraries()
    {
        var directory = Path.GetDirectoryName(typeof(NDSharpKernel).Assembly.Location)!;
        var result = new List<Assembly>();
        foreach (var path in Directory.GetFiles(directory, "*.dll"))
        {
            var name = Path.GetFileName(path);
            bool library = name.StartsWith("NDSharp", StringComparison.OrdinalIgnoreCase) || name.StartsWith("MathemaSharp.", StringComparison.OrdinalIgnoreCase);
            if (!library || name.Contains("JupyterSharp", StringComparison.OrdinalIgnoreCase)) continue; // the plugin itself is not a library for cells
            try { result.Add(Assembly.LoadFrom(path)); } catch (BadImageFormatException) { /* not a managed library */ }
        }
        return result;
    }

    public Task ExecuteAsync(string code, IKernelOutputSink sink, CancellationToken cancellationToken)
        => _inner.ExecuteAsync(code, sink, cancellationToken);

    public Task SetVariableAsync(string name, object? value, CancellationToken cancellationToken)
        => _inner.SetVariableAsync(name, value, cancellationToken);
}
