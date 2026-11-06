// Copyright (c) 2026 Marco Parenzan
//
// Licensed under the MIT License. See the LICENSE file in the project
// root for full license information.

using System.Numerics;
using MathemaSharp.Kernels;

namespace NDSharp;

/// <summary>Axes helpers of the reductions. The reductions themselves run on the lane kernels of MathemaSharp.Kernels (see <see cref="KernelPath"/>); the general
/// engine that used to fold arrays here was removed on 2026-11-06 (git tag <c>engine-before-removal-20261106</c> in NDSharp keeps it).</summary>
internal static class Red
{
    // ------------------------------------------------------------------ axes handling

    public static int[] NormalizeAxes(int[]? axes, int ndim)
    {
        if (axes is null) return Enumerable.Range(0, ndim).ToArray();
        var r = axes.Select(a => np.NormalizeAxis(a, ndim)).ToArray();
        if (r.Distinct().Count() != r.Length) throw new NDValueException("duplicate value in 'axis'");
        return r.OrderBy(x => x).ToArray();
    }

    public static int[] OutShape(int[] shape, int[] axes, bool keepdims)
    {
        var l = new List<int>();
        for (int i = 0; i < shape.Length; i++)
        {
            if (axes.Contains(i)) { if (keepdims) l.Add(1); }
            else l.Add(shape[i]);
        }
        return l.ToArray();
    }
}

public static partial class np
{
    // ================================================================ sum / prod / min / max

    private static DType ReductionType(NDArray a, DType? dtype) => dtype ?? DTypes.SumResult(a.DType);

    // sum, prod, max, min, mean, var, std, cumsum and cumprod run on the lane kernels of MathemaSharp.Kernels (KernelPath) for every real dtype; complex
    // sums and means go through their real and imaginary parts, the temporal dtypes through their integer storage, and the other complex reductions are refused.

    private static NDNotSupportedException NotSupportedFor(DType dtype, string what)
        => new(dtype.IsComplex() ? "this reduction is not implemented for complex arrays" : $"{what} is not implemented for {dtype}");

    public static NDArray Sum(NDArray a, int[]? axis = null, bool keepdims = false, DType? dtype = null)
    {
        if (a.DType.IsTimeDelta()) return Temporal.Reduce(a, x => Sum(x, axis, keepdims));
        if (a.DType.IsDateTime()) throw new NDTypeException("ufunc 'add' cannot use operands with types dtype('" + a.DType.Name() + "') and dtype('" + a.DType.Name() + "')");
        if (a.DType.IsComplex()) return Cx.FromParts(Sum(Real(a), axis, keepdims), Sum(Imag(a), axis, keepdims), a.DType);
        var src = a.CastTo(ReductionType(a, dtype));
        var axes = Red.NormalizeAxes(axis, a.Ndim);
        return KernelPath.TryReduce(KernelPath.Op.Sum, src, axes, keepdims, 0, out var fast) ? fast : throw NotSupportedFor(src.DType, "sum");
    }

    public static NDArray Prod(NDArray a, int[]? axis = null, bool keepdims = false, DType? dtype = null)
    {
        var src = a.CastTo(ReductionType(a, dtype));
        var axes = Red.NormalizeAxes(axis, a.Ndim);
        return KernelPath.TryReduce(KernelPath.Op.Prod, src, axes, keepdims, 0, out var fast) ? fast : throw NotSupportedFor(src.DType, "prod");
    }

    public static NDArray Max(NDArray a, int[]? axis = null, bool keepdims = false)
        => a.DType.IsTemporal() ? Temporal.Reduce(a, x => Max(x, axis, keepdims)) : Extremum(a, axis, keepdims, KernelPath.Op.Max);

    public static NDArray Min(NDArray a, int[]? axis = null, bool keepdims = false)
        => a.DType.IsTemporal() ? Temporal.Reduce(a, x => Min(x, axis, keepdims)) : Extremum(a, axis, keepdims, KernelPath.Op.Min);

    private static NDArray Extremum(NDArray a, int[]? axis, bool keepdims, KernelPath.Op op)
    {
        var axes = Red.NormalizeAxes(axis, a.Ndim);
        bool isBool = a.DType == DType.Bool;
        if (KernelPath.TryReduce(op, isBool ? a.CastTo(DType.UInt8) : a, axes, keepdims, 0, out var fast)) return isBool ? fast.CastTo(DType.Bool) : fast;
        throw NotSupportedFor(a.DType, op == KernelPath.Op.Max ? "max" : "min");
    }

    /// <summary>Peak-to-peak (max − min).</summary>
    public static NDArray Ptp(NDArray a, int[]? axis = null, bool keepdims = false)
        => Subtract(Max(a, axis, keepdims), Min(a, axis, keepdims));

    public static NDArray Any(NDArray a, int[]? axis = null, bool keepdims = false)
    {
        var axes = Red.NormalizeAxes(axis, a.Ndim);
        if (KernelPath.TryTruth(false, a, axes, keepdims, out var fast)) return fast;
        return KernelPath.TryTruth(false, a.CastTo(DType.Bool), axes, keepdims, out var viaBool) ? viaBool : throw NotSupportedFor(a.DType, "any");
    }

    public static NDArray All(NDArray a, int[]? axis = null, bool keepdims = false)
    {
        var axes = Red.NormalizeAxes(axis, a.Ndim);
        if (KernelPath.TryTruth(true, a, axes, keepdims, out var fast)) return fast;
        return KernelPath.TryTruth(true, a.CastTo(DType.Bool), axes, keepdims, out var viaBool) ? viaBool : throw NotSupportedFor(a.DType, "all");
    }

    public static NDArray ArgMax(NDArray a, int? axis = null, bool keepdims = false) => ArgExtreme(a, axis, keepdims, true);
    public static NDArray ArgMin(NDArray a, int? axis = null, bool keepdims = false) => ArgExtreme(a, axis, keepdims, false);

    private static NDArray ArgExtreme(NDArray a, int? axis, bool keepdims, bool max)
    {
        int[] axes = axis is null ? Enumerable.Range(0, a.Ndim).ToArray() : new[] { NormalizeAxis(axis.Value, a.Ndim) };
        return KernelPath.TryArg(max, a, axes, keepdims, NanPolicy.Propagate, out var fast, out _) ? fast : throw NotSupportedFor(a.DType, max ? "argmax" : "argmin");
    }

    public static NDArray CumSum(NDArray a, int? axis = null, DType? dtype = null) => Cumulative(a, axis, dtype, KernelPath.Op.CumSum);
    public static NDArray CumProd(NDArray a, int? axis = null, DType? dtype = null) => Cumulative(a, axis, dtype, KernelPath.Op.CumProd);

    private static NDArray Cumulative(NDArray a, int? axis, DType? dtype, KernelPath.Op op)
    {
        var rt = dtype ?? DTypes.SumResult(a.DType);
        var src = a.CastTo(rt);
        int ax = axis is null ? 0 : NormalizeAxis(axis.Value, a.Ndim);   // checks the axis before anything else
        if (KernelPath.TryScan(op, src, axis is null ? null : ax, out var fast)) return fast;
        throw new NDNotSupportedException($"cumulative op on {rt}");
    }

    // ================================================================ mean / var / std

    private static NDArray CountOf(NDArray a, int[] axes, DType dt)
    {
        long n = 1;
        foreach (var ax in axes) n *= a.Shape[ax];
        return NDArray.Scalar((double)n).CastTo(dt);
    }

    public static NDArray Mean(NDArray a, int[]? axis = null, bool keepdims = false, DType? dtype = null)
    {
        var axes = Red.NormalizeAxes(axis, a.Ndim);
        if (a.DType.IsComplex())
            return Cx.FromParts(Mean(Real(a), axis, keepdims), Mean(Imag(a), axis, keepdims), a.DType);
        var dt = dtype ?? DTypes.MeanResult(a.DType);
        var src = a.CastTo(dt);
        if (KernelPath.TryReduce(KernelPath.Op.Mean, src, axes, keepdims, 0, out var fast)) return fast;
        // a dtype= that is not a float (the mean is summed in it and then divided): sum in that dtype, then divide
        if (!KernelPath.TryReduce(KernelPath.Op.Sum, src, axes, keepdims, 0, out var sum)) throw NotSupportedFor(dt, "mean");
        return Ew.Float<DivF>(sum, CountOf(a, axes, dt), dt);
    }

    /// <summary>numpy <c>var</c> as numpy computes it: mean, centred, squared, summed, divided by n − ddof (the kernel's two-pass variance).</summary>
    public static NDArray Var(NDArray a, int[]? axis = null, bool keepdims = false, int ddof = 0)
    {
        var axes = Red.NormalizeAxes(axis, a.Ndim);
        var dt = DTypes.MeanResult(a.DType);
        var af = a.CastTo(dt);
        return KernelPath.TryReduce(KernelPath.Op.Var, af, axes, keepdims, ddof, out var fast) ? fast : throw NotSupportedFor(dt, "var");
    }

    public static NDArray Std(NDArray a, int[]? axis = null, bool keepdims = false, int ddof = 0)
        => Sqrt(Var(a, axis, keepdims, ddof));

    // ================================================================ counting / searching

    public static NDArray CountNonzero(NDArray a, int[]? axis = null, bool keepdims = false)
        => KernelPath.TryCountNonzero(a, Red.NormalizeAxes(axis, a.Ndim), keepdims, out var fast) ? fast : Sum(NotEqual(a, NDArray.WeakScalar(0L)), axis, keepdims);

    /// <summary>numpy <c>nonzero</c>: one int64 index array per dimension, C-order.</summary>
    public static NDArray[] NonZero(NDArray a)
    {
        var b = a.CastTo(DType.Bool);
        if (b.Ndim == 0) b = Reshape(b, 1);
        var mask = (bool[])Cast.ToBuffer(b, DType.Bool);
        int count = 0;
        foreach (var v in mask) if (v) count++;
        var outs = Enumerable.Range(0, b.Ndim).Select(_ => new long[count]).ToArray();
        var idx = new int[b.Ndim];
        int k = 0;
        for (int flat = 0; flat < mask.Length; flat++)
        {
            if (mask[flat])
            {
                for (int d = 0; d < b.Ndim; d++) outs[d][k] = idx[d];
                k++;
            }
            for (int d = b.Ndim - 1; d >= 0; d--)
            {
                if (++idx[d] < b.Shape[d]) break;
                idx[d] = 0;
            }
        }
        return outs.Select(o => NDArray.FromArray(o, count)).ToArray();
    }
}
