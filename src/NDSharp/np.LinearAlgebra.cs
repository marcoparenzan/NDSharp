// Copyright (c) 2026 Marco Parenzan
//
// Licensed under the MIT License. See the LICENSE file in the project
// root for full license information.

using MathemaSharp.LinearAlgebra;

namespace NDSharp;

/// <summary><c>numpy.linalg</c>: dense solvers and decompositions. All maths is done in double and the
/// result rounded back to float32 for float32 input (numpy's own LAPACK would compute in float32; the
/// difference is at rounding level). Stacked matrices (leading batch dimensions) are supported.</summary>
public static partial class np
{
    private static DType LinalgType(NDArray a)
    {
        if (a.DType is DType.Float32) return DType.Float32;
        if (a.DType is DType.Float64 || !a.DType.IsFloat()) return DType.Float64;
        return DType.Float32; // float16 inputs are promoted to float32 by numpy; we stay close enough
    }

    private static void RequireSquareStack(NDArray a, string name)
    {
        if (a.Ndim < 2)
            throw new NDLinAlgException($"{a.Ndim}-dimensional array given. Array must be at least two-dimensional");
        if (a.Shape[^1] != a.Shape[^2])
            throw new NDLinAlgException("Last 2 dimensions of the array must be square");
    }

    /// <summary>The matrices of a stack as row-major double arrays.</summary>
    private static (double[][] mats, int[] batch, int r, int c) Matrices(NDArray a)
    {
        int r = a.Shape[^2], c = a.Shape[^1];
        var batch = a.Shape[..^2];
        var flat = a.AsType(DType.Float64).ToArray<double>();
        int count = NDArray.SizeOf(batch);
        var mats = new double[count][];
        for (int i = 0; i < count; i++)
        {
            mats[i] = new double[r * c];
            Array.Copy(flat, i * r * c, mats[i], 0, r * c);
        }
        return (mats, batch, r, c);
    }

    private static NDArray FromMatrices(double[][] mats, int[] batch, int r, int c, DType dt)
    {
        var flat = new double[mats.Length * r * c];
        for (int i = 0; i < mats.Length; i++) Array.Copy(mats[i], 0, flat, i * r * c, r * c);
        var arr = new NDArray(DType.Float64, flat, batch.Concat(new[] { r, c }).ToArray());
        return dt == DType.Float64 ? arr : arr.AsType(dt);
    }

    public static NDArray Inv(NDArray a)
    {
        RequireSquareStack(a, "inv");
        var dt = LinalgType(a);
        var (mats, batch, n, _) = Matrices(a);
        var result = new double[mats.Length][];
        for (int t = 0; t < mats.Length; t++)
        {
            var lu = (double[])mats[t].Clone();
            var piv = new int[n];
            if (DenseLinearAlgebra.Lu(lu, n, piv)) throw new NDLinAlgException("Singular matrix");
            var inv = new double[n * n];
            for (int i = 0; i < n; i++) inv[i * n + i] = 1.0;
            DenseLinearAlgebra.LuSolve(lu, n, piv, inv, n);
            result[t] = inv;
        }
        return FromMatrices(result, batch, n, n, dt);
    }

    /// <summary>numpy <c>linalg.solve(a, b)</c>; <paramref name="b"/> is a vector (n) or matrix (n, k).</summary>
    public static NDArray Solve(NDArray a, NDArray b)
    {
        RequireSquareStack(a, "solve");
        var dt = LinalgType(a);
        int n = a.Shape[^1];
        bool vec = b.Ndim == 1 || (b.Ndim == a.Ndim - 1 && b.Shape[^1] == n && a.Ndim > 2);
        var bm = vec ? ExpandDims(b, -1) : b;
        if (bm.Shape[^2] != n)
            throw new NDValueException("solve: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (m,m),(m,n)->(m,n) (size " + bm.Shape[^2] + " is different from " + n + ")");
        var (am, batchA, _, _) = Matrices(a);
        var batchShape = Broadcasting.Shape(batchA, bm.Shape[..^2]);
        int count = NDArray.SizeOf(batchShape);
        var aFull = Broadcasting.BroadcastTo(a.AsType(DType.Float64), batchShape.Concat(new[] { n, n }).ToArray()).Copy().ToArray<double>();
        int k = bm.Shape[^1];
        var bFull = Broadcasting.BroadcastTo(bm.AsType(DType.Float64), batchShape.Concat(new[] { n, k }).ToArray()).Copy().ToArray<double>();
        for (int t = 0; t < count; t++)
        {
            var lu = new double[n * n];
            Array.Copy(aFull, t * n * n, lu, 0, n * n);
            var piv = new int[n];
            if (DenseLinearAlgebra.Lu(lu, n, piv)) throw new NDLinAlgException("Singular matrix");
            var rhs = new double[n * k];
            Array.Copy(bFull, t * n * k, rhs, 0, n * k);
            DenseLinearAlgebra.LuSolve(lu, n, piv, rhs, k);
            Array.Copy(rhs, 0, bFull, t * n * k, n * k);
        }
        var res = new NDArray(DType.Float64, bFull, batchShape.Concat(new[] { n, k }).ToArray());
        if (vec) res = Squeeze(res, res.Ndim - 1);
        return dt == DType.Float64 ? res : res.AsType(dt);
    }

    /// <summary>numpy <c>linalg.slogdet</c>: (sign, log|det|).</summary>
    public static (NDArray sign, NDArray logdet) SlogDet(NDArray a)
    {
        RequireSquareStack(a, "slogdet");
        var dt = LinalgType(a);
        var (mats, batch, n, _) = Matrices(a);
        var signs = new double[mats.Length];
        var logs = new double[mats.Length];
        for (int t = 0; t < mats.Length; t++)
        {
            (signs[t], logs[t]) = DenseLinearAlgebra.SignAndLogDeterminant(mats[t], n);
        }
        return (new NDArray(dt == DType.Float64 ? DType.Float64 : dt, dt == DType.Float64 ? signs : signs.Select(x => (float)x).ToArray(), (int[])batch.Clone()),
                new NDArray(dt == DType.Float64 ? DType.Float64 : dt, dt == DType.Float64 ? logs : logs.Select(x => (float)x).ToArray(), (int[])batch.Clone()));
    }

    /// <summary>numpy <c>linalg.det</c>, computed as numpy does: <c>sign * exp(sum(log|u_ii|))</c>.</summary>
    public static NDArray Det(NDArray a)
    {
        var (sign, logdet) = SlogDet(a);
        var dt = sign.DType;
        var d = Multiply(sign, Exp(logdet));
        return d.DType == dt ? d : d.AsType(dt);
    }

    /// <summary>numpy <c>linalg.eigh</c> (lower triangle is used, as numpy's default UPLO='L').</summary>
    public static (NDArray w, NDArray v) Eigh(NDArray a)
    {
        RequireSquareStack(a, "eigh");
        var dt = LinalgType(a);
        var (mats, batch, n, _) = Matrices(a);
        var ws = new double[mats.Length][];
        var vs = new double[mats.Length][];
        for (int t = 0; t < mats.Length; t++)
        {
            var sym = (double[])mats[t].Clone();
            for (int i = 0; i < n; i++)
                for (int j = i + 1; j < n; j++) sym[i * n + j] = sym[j * n + i]; // mirror the lower triangle
            (ws[t], vs[t]) = DenseLinearAlgebra.SymmetricEigen(sym, n);
        }
        var w = FromMatrices(ws, batch, 1, n, dt);
        w = Reshape(w, batch.Concat(new[] { n }).ToArray());
        return (w, FromMatrices(vs, batch, n, n, dt));
    }

    public static NDArray EigvalsH(NDArray a) => Eigh(a).w;

    public sealed record SvdResult(NDArray U, NDArray S, NDArray Vt);

    /// <summary>numpy <c>linalg.svd</c>. Singular values match numpy; the signs of singular vectors may differ from LAPACK's.</summary>
    public static SvdResult Svd(NDArray a, bool fullMatrices = true)
    {
        if (a.Ndim < 2) throw new NDLinAlgException($"{a.Ndim}-dimensional array given. Array must be at least two-dimensional");
        var dt = LinalgType(a);
        var (mats, batch, rows, cols) = Matrices(a);
        int k = Math.Min(rows, cols);
        var us = new double[mats.Length][];
        var vts = new double[mats.Length][];
        var ss = new double[mats.Length][];
        int uc = fullMatrices ? rows : k, vr = fullMatrices ? cols : k;
        for (int t = 0; t < mats.Length; t++)
        {
            double[] u, s, v;
            if (rows >= cols)
            {
                (u, s, v) = DenseLinearAlgebra.ThinSvd(mats[t], rows, cols);   // u: rows×cols, v: cols×cols
            }
            else
            {
                var tr = new double[rows * cols];
                for (int i = 0; i < rows; i++) for (int j = 0; j < cols; j++) tr[j * rows + i] = mats[t][i * cols + j];
                var (u2, s2, v2) = DenseLinearAlgebra.ThinSvd(tr, cols, rows); // tr = U2 S V2^T  =>  a = V2 S U2^T
                u = v2; s = s2; v = u2;                       // u: rows×rows, v: cols×rows
            }
            // LAPACK (Householder bidiagonalization) leaves the leading singular vectors with a negative first
            // component far more often than not; mimic that for the dominant triplet so results agree more often.
            if (k > 0 && v[0] > 1e-12)
            {
                int vc = rows >= cols ? cols : rows, uc2 = rows >= cols ? cols : rows;
                for (int i = 0; i < (rows >= cols ? rows : cols); i++) { if (rows >= cols) u[i * uc2] = -u[i * uc2]; else v[i * vc] = -v[i * vc]; }
                for (int i = 0; i < (rows >= cols ? cols : rows); i++) { if (rows >= cols) v[i * vc] = -v[i * vc]; else u[i * uc2] = -u[i * uc2]; }
            }
            double[] uFull = fullMatrices && rows > k ? DenseLinearAlgebra.CompleteBasis(u, rows, k) : u;
            double[] vFull = fullMatrices && cols > k ? DenseLinearAlgebra.CompleteBasis(v, cols, k) : v;
            int ucols = fullMatrices && rows > k ? rows : k;
            int vcols = fullMatrices && cols > k ? cols : k;
            us[t] = uFull;
            ss[t] = s;
            var vt = new double[vcols * cols];
            for (int i = 0; i < vcols; i++) for (int j = 0; j < cols; j++) vt[i * cols + j] = vFull[j * vcols + i];
            vts[t] = vt;
            _ = ucols;
        }
        var U = FromMatrices(us, batch, rows, uc, dt);
        var Vt = FromMatrices(vts, batch, vr, cols, dt);
        var S = FromMatrices(ss, batch, 1, k, dt);
        return new SvdResult(U, Reshape(S, batch.Concat(new[] { k }).ToArray()), Vt);
    }

    public static NDArray SingularValues(NDArray a) => Svd(a, false).S;

    public static int MatrixRank(NDArray a, double? tol = null)
    {
        var s = SingularValues(a).AsType(DType.Float64).ToArray<double>();
        if (s.Length == 0) return 0;
        double t = tol ?? s.Max() * Math.Max(a.Shape[^2], a.Shape[^1]) * 2.220446049250313e-16;
        return s.Count(x => x > t);
    }

    /// <summary>numpy <c>linalg.lstsq</c>: minimum-norm least-squares solution, residuals, rank and singular values.</summary>
    public static (NDArray x, NDArray residuals, int rank, NDArray s) LstSq(NDArray a, NDArray b, double? rcond = null)
    {
        if (a.Ndim != 2) throw new NDLinAlgException($"{a.Ndim}-dimensional array given. Array must be two-dimensional");
        bool vec = b.Ndim == 1;
        var bm = vec ? ExpandDims(b, -1) : b;
        int m = a.Shape[0], n = a.Shape[1];
        if (bm.Shape[0] != m) throw new NDLinAlgException("Incompatible dimensions");
        var dt = LinalgType(a);
        var svd = Svd(a, false);
        var s = svd.S.AsType(DType.Float64).ToArray<double>();
        double cutoff = (rcond ?? 2.220446049250313e-16 * Math.Max(m, n)) * (s.Length == 0 ? 0 : s.Max());
        int rank = s.Count(x => x > cutoff);
        var U = svd.U.AsType(DType.Float64);
        var Vt = svd.Vt.AsType(DType.Float64);
        // x = V * diag(1/s) * U^T * b  (only the retained singular values)
        var utb = MatMul(Transpose(U), bm.AsType(DType.Float64));
        var scale = NDArray.FromArray(s.Select(x => x > cutoff ? 1.0 / x : 0.0).ToArray());
        var x = MatMul(Transpose(Vt), Multiply(ExpandDims(scale, 1), utb));
        NDArray residuals;
        if (rank == n && m > n)
        {
            var r = Subtract(bm.AsType(DType.Float64), MatMul(a.AsType(DType.Float64), x));
            residuals = Sum(Multiply(r, r), new[] { 0 });
        }
        else residuals = np.Zeros(new[] { 0 }, DType.Float64);
        if (vec) { x = Squeeze(x, 1); }
        return (dt == DType.Float64 ? x : x.AsType(dt), dt == DType.Float64 || residuals.Size == 0 ? residuals : residuals.AsType(dt), rank, svd.S);
    }

    public static NDArray Pinv(NDArray a, double? rcond = null)
    {
        var svd = Svd(a, false);
        var s = svd.S.AsType(DType.Float64).ToArray<double>();
        double cutoff = (rcond ?? 1e-15) * (s.Length == 0 ? 0 : s.Max());
        var inv = NDArray.FromArray(s.Select(x => x > cutoff ? 1.0 / x : 0.0).ToArray());
        return MatMul(Multiply(Transpose(svd.Vt.AsType(DType.Float64)), inv), Transpose(svd.U.AsType(DType.Float64)));
    }
}
