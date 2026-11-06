using JupyterSharp.Kernels.Abstractions;

namespace NDSharp.JupyterSharp.Kernel;

/// <summary>The entire surface JupyterSharp.Host needs to load this plugin — see IKernelPlugin.</summary>
public sealed class NDSharpKernelPlugin : IKernelPlugin
{
    public string KernelId => "ndsharp";

    public IKernel CreateKernel(INotebookHost notebookHost) => new NDSharpKernel();
}
