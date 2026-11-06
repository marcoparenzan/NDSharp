// Copyright (c) 2026 Marco Parenzan
//
// Licensed under the MIT License. See the LICENSE file in the project
// root for full license information.

using MathemaSharp.Kernels;

namespace NDSharp;

public static partial class np
{
    // ================================================================ nan* functions (on MathemaSharp.Kernels)
    // numpy's nansum, nanprod, nanmean, nanvar, nanstd, nanmax, nanmin, nanmedian, nanpercentile, nanquantile, nanargmax, nanargmin, nancumsum and nancumprod.
    // Floating-point input runs on the kernels with NaN omitted, with numpy's rules: a NaN is replaced by the identity before summing, an all-NaN slice gives
    // NaN for mean, var, std, max, min, median and the percentiles (numpy also warns), and zero or one for sum and product. Integers and bool have no NaN:
    // they are handed to the plain function, as numpy does. Complex input is not implemented.

    private static bool IsNanAware(NDArray a, string name)
    {
        if (a.DType.IsComplex()) throw new NDNotSupportedException($"{name} is not implemented for complex arrays");
        return a.DType.IsFloat();
    }

    private static NDArray NanReduce(KernelPath.Op op, NDArray src, int[] axes, bool keepdims, int ddof = 0)
        => KernelPath.TryReduce(op, src, axes, keepdims, ddof, NanPolicy.Omit, out var r) ? r : throw new NDNotSupportedException($"nan-aware {op} is not implemented for {src.DType}");

    /// <summary>numpy <c>nansum</c>: the sum with NaN treated as zero.</summary>
    public static NDArray NanSum(NDArray a, int[]? axis = null, bool keepdims = false, DType? dtype = null)
        => IsNanAware(a, "nansum") ? NanReduce(KernelPath.Op.Sum, a.CastTo(ReductionType(a, dtype)), Red.NormalizeAxes(axis, a.Ndim), keepdims) : Sum(a, axis, keepdims, dtype);

    /// <summary>numpy <c>nanprod</c>: the product with NaN treated as one.</summary>
    public static NDArray NanProd(NDArray a, int[]? axis = null, bool keepdims = false, DType? dtype = null)
        => IsNanAware(a, "nanprod") ? NanReduce(KernelPath.Op.Prod, a.CastTo(ReductionType(a, dtype)), Red.NormalizeAxes(axis, a.Ndim), keepdims) : Prod(a, axis, keepdims, dtype);

    /// <summary>numpy <c>nanmean</c>: the mean of the values that are not NaN; NaN for a slice with none.</summary>
    public static NDArray NanMean(NDArray a, int[]? axis = null, bool keepdims = false, DType? dtype = null)
        => IsNanAware(a, "nanmean") ? NanReduce(KernelPath.Op.Mean, a.CastTo(dtype ?? DTypes.MeanResult(a.DType)), Red.NormalizeAxes(axis, a.Ndim), keepdims) : Mean(a, axis, keepdims, dtype);

    /// <summary>numpy <c>nanvar</c>: the variance of the values that are not NaN, divided by n − ddof; NaN when that is not positive.</summary>
    public static NDArray NanVar(NDArray a, int[]? axis = null, bool keepdims = false, int ddof = 0)
        => IsNanAware(a, "nanvar") ? NanReduce(KernelPath.Op.Var, a, Red.NormalizeAxes(axis, a.Ndim), keepdims, ddof) : Var(a, axis, keepdims, ddof);

    /// <summary>numpy <c>nanstd</c>: the square root of <see cref="NanVar"/>.</summary>
    public static NDArray NanStd(NDArray a, int[]? axis = null, bool keepdims = false, int ddof = 0)
        => IsNanAware(a, "nanstd") ? Sqrt(NanVar(a, axis, keepdims, ddof)) : Std(a, axis, keepdims, ddof);

    /// <summary>numpy <c>nanmax</c>: the largest value that is not NaN; NaN for an all-NaN slice.</summary>
    public static NDArray NanMax(NDArray a, int[]? axis = null, bool keepdims = false)
        => IsNanAware(a, "nanmax") ? NanReduce(KernelPath.Op.Max, a, Red.NormalizeAxes(axis, a.Ndim), keepdims) : Max(a, axis, keepdims);

    /// <summary>numpy <c>nanmin</c>: the smallest value that is not NaN; NaN for an all-NaN slice.</summary>
    public static NDArray NanMin(NDArray a, int[]? axis = null, bool keepdims = false)
        => IsNanAware(a, "nanmin") ? NanReduce(KernelPath.Op.Min, a, Red.NormalizeAxes(axis, a.Ndim), keepdims) : Min(a, axis, keepdims);

    /// <summary>numpy <c>nanmedian</c>: the median of the values that are not NaN; NaN for an all-NaN slice.</summary>
    public static NDArray NanMedian(NDArray a, int[]? axis = null, bool keepdims = false)
    {
        if (!IsNanAware(a, "nanmedian")) return Median(a, axis, keepdims);
        return KernelPath.TryOrder(a, Red.NormalizeAxes(axis, a.Ndim), keepdims, null, default, NanPolicy.Omit, out var r) ? r : throw new NDNotSupportedException($"nanmedian is not implemented for {a.DType}");
    }

    /// <summary>numpy <c>nanpercentile</c> / <c>nanquantile</c> (<paramref name="isQuantile"/>): <see cref="Percentile(NDArray, NDArray, int[], bool, bool, string)"/> over the values that are not NaN.</summary>
    public static NDArray NanPercentile(NDArray a, NDArray q, int[]? axis = null, bool keepdims = false, bool isQuantile = false, string method = "linear")
        => IsNanAware(a, "nanpercentile") ? PercentileCore(a, q, axis, keepdims, isQuantile, method, NanPolicy.Omit) : Percentile(a, q, axis, keepdims, isQuantile, method);

    /// <summary>numpy <c>nanargmax</c>: the index of the largest value that is not NaN; fails for a slice that is all NaN.</summary>
    public static NDArray NanArgMax(NDArray a, int? axis = null, bool keepdims = false) => NanArg(a, axis, keepdims, true);

    /// <summary>numpy <c>nanargmin</c>: the index of the smallest value that is not NaN; fails for a slice that is all NaN.</summary>
    public static NDArray NanArgMin(NDArray a, int? axis = null, bool keepdims = false) => NanArg(a, axis, keepdims, false);

    private static NDArray NanArg(NDArray a, int? axis, bool keepdims, bool max)
    {
        if (!IsNanAware(a, max ? "nanargmax" : "nanargmin")) return max ? ArgMax(a, axis, keepdims) : ArgMin(a, axis, keepdims);
        int[] axes = axis is null ? Enumerable.Range(0, a.Ndim).ToArray() : new[] { NormalizeAxis(axis.Value, a.Ndim) };
        if (!KernelPath.TryArg(max, a, axes, keepdims, NanPolicy.Omit, out var r, out bool allNan)) throw new NDNotSupportedException($"nan-aware arg-extreme is not implemented for {a.DType}");
        if (allNan) throw new NDValueException("All-NaN slice encountered");
        return r;
    }

    /// <summary>numpy <c>nancumsum</c>: the running sum with NaN treated as zero.</summary>
    public static NDArray NanCumSum(NDArray a, int? axis = null, DType? dtype = null) => NanCumulative(a, axis, dtype, KernelPath.Op.CumSum);

    /// <summary>numpy <c>nancumprod</c>: the running product with NaN treated as one.</summary>
    public static NDArray NanCumProd(NDArray a, int? axis = null, DType? dtype = null) => NanCumulative(a, axis, dtype, KernelPath.Op.CumProd);

    private static NDArray NanCumulative(NDArray a, int? axis, DType? dtype, KernelPath.Op op)
    {
        if (!IsNanAware(a, op == KernelPath.Op.CumSum ? "nancumsum" : "nancumprod")) return op == KernelPath.Op.CumSum ? CumSum(a, axis, dtype) : CumProd(a, axis, dtype);
        var src = a.CastTo(dtype ?? DTypes.SumResult(a.DType));
        int? ax = axis is null ? null : NormalizeAxis(axis.Value, a.Ndim);
        return KernelPath.TryScan(op, src, ax, NanPolicy.Omit, out var r) ? r : throw new NDNotSupportedException($"nan-aware cumulative operation is not implemented for {src.DType}");
    }
}
