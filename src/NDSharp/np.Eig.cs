using System.Numerics;

namespace NDSharp;

public static partial class np
{
    /// <summary>
    /// numpy <c>linalg.eig</c>: eigenvalues <c>w</c> and right eigenvectors <c>v</c> (column <c>v[..., :, i]</c> belongs to <c>w[..., i]</c>) of a
    /// general square matrix or stack of matrices. Each vector has unit norm. The result is always complex (complex128, or complex64 for single-precision input) —
    /// numpy ≥ 2.5 no longer narrows it to real when every eigenvalue is real; callers that want a real result check the imaginary
    /// parts themselves. The order of the eigenvalues is not specified.
    /// </summary>
    public static (NDArray w, NDArray v) Eig(NDArray a)
    {
        var (w, v) = EigCore(a, vectors: true);
        return (w, v!);
    }

    /// <summary>numpy <c>linalg.eigvals</c>: the eigenvalues of <see cref="Eig"/> without the eigenvectors.</summary>
    public static NDArray Eigvals(NDArray a) => EigCore(a, vectors: false).w;

    private static (NDArray w, NDArray? v) EigCore(NDArray a, bool vectors)
    {
        string name = vectors ? "eig" : "eigvals";
        RequireSquareStack(a, name);
        int n = a.Shape[^1];
        var batch = a.Shape[..^2];
        int count = NDArray.SizeOf(batch);
        bool complexIn = a.DType.IsComplex();
        bool single = a.DType is DType.Float32 or DType.Complex64;

        var flat = complexIn ? a.AsType(DType.Complex128).ToArray<Complex>() : a.AsType(DType.Float64).ToArray<double>().Select(x => new Complex(x, 0)).ToArray();
        foreach (var z in flat)
            if (!double.IsFinite(z.Real) || !double.IsFinite(z.Imaginary))
                throw new NDLinAlgException("Array must not contain infs or NaNs");

        var ws = new Complex[count * n];
        var vs = vectors ? new Complex[count * n * n] : null;
        for (int t = 0; t < count; t++)
        {
            var mat = new Complex[n * n];
            Array.Copy(flat, t * n * n, mat, 0, n * n);
            var (w, v) = EigSolver.Solve(mat, n, vectors, realInput: !complexIn);
            Array.Copy(w, 0, ws, t * n, n);
            if (vs != null) Array.Copy(v!, 0, vs, t * n * n, n * n);
        }

        var wShape = batch.Concat(new[] { n }).ToArray();
        var vShape = batch.Concat(new[] { n, n }).ToArray();
        NDArray W = new NDArray(DType.Complex128, ws, wShape);
        NDArray? V = vs == null ? null : new NDArray(DType.Complex128, vs, vShape);
        if (single)
        {
            W = W.AsType(DType.Complex64);
            V = V?.AsType(DType.Complex64);
        }
        return (W, V);
    }
}


