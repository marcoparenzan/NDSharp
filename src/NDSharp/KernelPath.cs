// Copyright (c) 2026 Marco Parenzan
//
// Licensed under the MIT License. See the LICENSE file in the project
// root for full license information.

using System.Numerics;
using MathemaSharp.Kernels;

namespace NDSharp;

/// <summary>Runs <c>sum prod max min mean var std cumsum cumprod</c> on the lane kernels of <c>MathemaSharp.Kernels</c>, for every real dtype
/// (integers, float16, float32, float64), over any axes, any layout and empty arrays, with numpy's rules: NaN propagates, sums are
/// pairwise when the reduced elements are adjacent in memory (and plain from the first element otherwise), variance is two-pass. Float16 is computed in
/// float16 arithmetic, as the general engine did (NumPy accumulates float16 sums in float32: not verified here, open).
/// A C-contiguous array whose reduced axes are adjacent is read in place; any other layout is first gathered into a contiguous copy with the
/// reduced axes last. Complex and the temporal dtypes are left to the general engine.</summary>
public static class KernelPath
{
    static readonly AsyncLocal<Scope?> current = new();

    /// <summary>Starts a scope in the current execution context (it flows to the threads and tasks started from it) that switches the kernel path
    /// on or off and counts the calls it answers. Dispose it to restore the previous setting. Meant for tests and diagnostics; the default is on.
    /// With the path off, every call uses the general engine.</summary>
    public static Scope Begin(bool enabled)
    {
        var s = new Scope(enabled, current.Value);
        current.Value = s;
        return s;
    }

    /// <summary>A switch and a counter for the kernel path, local to an execution context.</summary>
    public sealed class Scope : IDisposable
    {
        readonly Scope? previous;
        long calls;
        internal Scope(bool enabled, Scope? previous) { Enabled = enabled; this.previous = previous; }
        /// <summary>Whether the kernel path is on inside this scope.</summary>
        public bool Enabled { get; }
        /// <summary>Calls answered by the kernels inside this scope.</summary>
        public long Calls => Interlocked.Read(ref calls);
        internal void Count() => Interlocked.Increment(ref calls);
        /// <inheritdoc/>
        public void Dispose() => current.Value = previous;
    }

    /// <summary>Whether the kernel path is on in the current execution context (it is, unless a scope switched it off).</summary>
    internal static bool Active => current.Value is not { Enabled: false };

    /// <summary>Counts one call answered by the kernels in the current scope (for the products, which do their own dispatch).</summary>
    internal static void CountCall() => current.Value?.Count();

    internal enum Op { Sum, Prod, Max, Min, Mean, Var, CumSum, CumProd }

    /// <summary>Tries the kernels. <paramref name="a"/> is already cast to the dtype of the computation; <paramref name="axes"/> is the normalised,
    /// sorted list of reduced axes (all of them when the caller passed none).</summary>
    internal static bool TryReduce(Op op, NDArray a, int[] axes, bool keepdims, int ddof, out NDArray result)
        => TryReduce(op, a, axes, keepdims, ddof, NanPolicy.Propagate, out result);

    /// <summary>As above with a NaN policy: <see cref="NanPolicy.Omit"/> serves the <c>nan*</c> functions, which have no general-engine implementation and so ignore <see cref="Begin"/>.</summary>
    internal static bool TryReduce(Op op, NDArray a, int[] axes, bool keepdims, int ddof, NanPolicy nan, out NDArray result)
    {
        result = null!;
        var scope = current.Value;
        if (scope is { Enabled: false } && nan == NanPolicy.Propagate) return false;
        NDArray? r = op is Op.Mean or Op.Var ? ReduceFloat(op, a, axes, keepdims, ddof, nan) : ReduceNumber(op, a, axes, keepdims, nan);
        if (r is null) return false;
        scope?.Count();
        result = r;
        return true;
    }

    /// <summary>Tries the kernels for <c>cumsum</c> and <c>cumprod</c>; <paramref name="a"/> is already cast; <paramref name="axis"/> is normalised, or null for the flattened array.</summary>
    internal static bool TryScan(Op op, NDArray a, int? axis, out NDArray result)
        => TryScan(op, a, axis, NanPolicy.Propagate, out result);

    internal static bool TryScan(Op op, NDArray a, int? axis, NanPolicy nan, out NDArray result)
    {
        result = null!;
        var scope = current.Value;
        if (scope is { Enabled: false } && nan == NanPolicy.Propagate) return false;
        NDArray? r = a.DType switch
        {
            DType.Int8 => Scan<sbyte>(op, a, axis, nan),
            DType.UInt8 => Scan<byte>(op, a, axis, nan),
            DType.Int16 => Scan<short>(op, a, axis, nan),
            DType.UInt16 => Scan<ushort>(op, a, axis, nan),
            DType.Int32 => Scan<int>(op, a, axis, nan),
            DType.UInt32 => Scan<uint>(op, a, axis, nan),
            DType.Int64 => Scan<long>(op, a, axis, nan),
            DType.UInt64 => Scan<ulong>(op, a, axis, nan),
            DType.Float16 => Scan<Half>(op, a, axis, nan),
            DType.Float32 => Scan<float>(op, a, axis, nan),
            DType.Float64 => Scan<double>(op, a, axis, nan),
            _ => null,
        };
        if (r is null) return false;
        scope?.Count();
        result = r;
        return true;
    }

    static NDArray? ReduceNumber(Op op, NDArray a, int[] axes, bool keepdims, NanPolicy nan) => a.DType switch
    {
        DType.Int8 => Reduce<sbyte>(op, a, axes, keepdims, nan),
        DType.UInt8 => Reduce<byte>(op, a, axes, keepdims, nan),
        DType.Int16 => Reduce<short>(op, a, axes, keepdims, nan),
        DType.UInt16 => Reduce<ushort>(op, a, axes, keepdims, nan),
        DType.Int32 => Reduce<int>(op, a, axes, keepdims, nan),
        DType.UInt32 => Reduce<uint>(op, a, axes, keepdims, nan),
        DType.Int64 => Reduce<long>(op, a, axes, keepdims, nan),
        DType.UInt64 => Reduce<ulong>(op, a, axes, keepdims, nan),
        DType.Float16 => Reduce<Half>(op, a, axes, keepdims, nan),
        DType.Float32 => Reduce<float>(op, a, axes, keepdims, nan),
        DType.Float64 => Reduce<double>(op, a, axes, keepdims, nan),
        _ => null,
    };

    static NDArray? ReduceFloat(Op op, NDArray a, int[] axes, bool keepdims, int ddof, NanPolicy nan) => a.DType switch
    {
        DType.Float16 => MeanOrVar<Half>(op, a, axes, keepdims, ddof, nan),
        DType.Float32 => MeanOrVar<float>(op, a, axes, keepdims, ddof, nan),
        DType.Float64 => MeanOrVar<double>(op, a, axes, keepdims, ddof, nan),
        _ => null,
    };

    // ------------------------------------------------------------------ layout

    /// <summary>The lane layout of a reduction over <paramref name="axes"/> and the buffer the kernel reads.</summary>
    static T[] Prepare<T>(NDArray a, int[] axes, out int offset, out Lane lane, out SumMethod method, out SumMethod deviationMethod) where T : unmanaged
    {
        var shape = (int[])a.Shape.Clone();
        var strides = a.Strides;
        var buffer = (T[])a.Buffer;
        int nd = shape.Length, size = a.Size;
        bool adjacent = true;
        for (int i = 1; i < axes.Length; i++) if (axes[i] != axes[i - 1] + 1) adjacent = false;

        if (IsCContiguous(shape, strides) && adjacent)
        {
            // Read in place. The kernel sums pairwise exactly when the lane elements are adjacent (Inner == 1), as numpy does.
            offset = size == 0 ? 0 : a.Offset;
            if (axes.Length == 0) lane = new Lane(1, 1, size);
            else
            {
                int i = axes[0], j = axes[^1];
                lane = new Lane(Product(shape, i, j + 1), Product(shape, j + 1, nd), Product(shape, 0, i));
            }
            method = deviationMethod = SumMethod.Pairwise;
            return buffer;
        }

        // Gather into a contiguous copy with the kept axes first and the reduced axes last (both in their original order).
        var kept = Enumerable.Range(0, nd).Where(d => !axes.Contains(d)).ToArray();
        var perm = kept.Concat(axes).ToArray();
        var pshape = perm.Select(d => shape[d]).ToArray();
        var pstrides = perm.Select(d => strides[d]).ToArray();
        var copy = new T[size];
        if (size > 0)
        {
            var idx = new int[nd];
            int pos = a.Offset;
            for (int k = 0; k < size; k++)
            {
                copy[k] = buffer[pos];
                for (int d = nd - 1; d >= 0; d--)
                {
                    idx[d]++; pos += pstrides[d];
                    if (idx[d] < pshape[d]) break;
                    pos -= pstrides[d] * pshape[d]; idx[d] = 0;
                }
            }
        }
        lane = new Lane(Product(pshape, kept.Length, nd), 1, Product(pshape, 0, kept.Length));
        offset = 0;
        // numpy adds pairwise only when the reduced axes form one run with stride 1 in the ORIGINAL array; otherwise left to right from the first element.
        bool pairwise = false;
        if (axes.Length > 0 && lane.Length > 0)
        {
            var redShape = axes.Select(d => shape[d]).ToArray();
            var redStrides = axes.Select(d => strides[d]).ToArray();
            var rw = new RowWalker(redShape, redStrides);
            pairwise = rw.InnerLen == lane.Length && rw.InnerStride0 == 1;
        }
        method = pairwise ? SumMethod.Pairwise : SumMethod.SequentialFromFirst;
        // The squared deviations of a variance are a new C-contiguous array of the original shape: pairwise only for adjacent reduced axes that end the shape.
        bool trailing = axes.Length == 0 || (adjacent && Product(shape, axes[^1] + 1, nd) == 1);
        deviationMethod = trailing ? SumMethod.Pairwise : SumMethod.SequentialFromFirst;
        return copy;
    }

    static bool IsCContiguous(int[] shape, int[] strides)
    {
        long expected = 1;
        for (int d = shape.Length - 1; d >= 0; d--)
        {
            if (shape[d] == 0) return true;
            if (shape[d] != 1 && strides[d] != expected) return false;
            expected *= shape[d];
        }
        return true;
    }

    static int Product(int[] shape, int from, int to)
    {
        long p = 1;
        for (int i = from; i < to; i++) p *= shape[i];
        return checked((int)p);
    }

    static ReadOnlySpan<T> Window<T>(T[] buffer, int offset, Lane lane) => lane.Total == 0 ? [] : buffer.AsSpan(offset, lane.Total);

    // ------------------------------------------------------------------ reductions

    static NDArray Reduce<T>(Op op, NDArray a, int[] axes, bool keepdims, NanPolicy nan) where T : unmanaged, INumber<T>
    {
        var buffer = Prepare<T>(a, axes, out int offset, out var lane, out var method, out _);
        var src = Window(buffer, offset, lane);
        var values = new T[lane.Count];
        switch (op)
        {
            case Op.Sum: Reductions.Sum<T>(src, lane, values, nan, method); break;
            case Op.Prod: Reductions.Product<T>(src, lane, values, nan); break;
            case Op.Max:
            case Op.Min:
                if (lane.Count > 0 && lane.Length == 0)
                    throw new NDValueException($"zero-size array to reduction operation {(op == Op.Max ? "maximum" : "minimum")} which has no identity");
                if (op == Op.Max) Reductions.Max<T>(src, lane, values, Span<int>.Empty, nan);
                else Reductions.Min<T>(src, lane, values, Span<int>.Empty, nan);
                break;
            default: throw new InvalidOperationException(op.ToString());
        }
        return new NDArray(a.DType, values, Red.OutShape(a.Shape, axes, keepdims));
    }

    static NDArray MeanOrVar<T>(Op op, NDArray a, int[] axes, bool keepdims, int ddof, NanPolicy nan) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var buffer = Prepare<T>(a, axes, out int offset, out var lane, out var method, out var deviationMethod);
        var src = Window(buffer, offset, lane);
        var values = new T[lane.Count];
        if (op == Op.Mean) Reductions.Mean<T>(src, lane, values, nan, method);
        else
        {
            // numpy divides by max(n - ddof, 0): a positive sum of squares over zero gives +inf, zero or NaN gives NaN (the kernel gives NaN for n - ddof <= 0).
            // With NaN omitted (nanvar) the kernel's NaN for n - ddof <= 0 is exactly numpy's.
            bool degenerate = nan == NanPolicy.Propagate && lane.Length - ddof <= 0;
            Reductions.Variance<T>(src, lane, values, degenerate ? 0 : Math.Max(ddof, 0), nan, VarianceMethod.TwoPass, method, deviationMethod);
            if (degenerate)
                for (int i = 0; i < values.Length; i++) values[i] = values[i] > T.Zero ? T.PositiveInfinity : T.NaN;
        }
        return new NDArray(a.DType, values, Red.OutShape(a.Shape, axes, keepdims));
    }

    // ------------------------------------------------------------------ order statistics

    /// <summary>Tries the kernels for <c>median</c> (<paramref name="q"/> null) or the quantiles <paramref name="q"/> (probabilities in [0, 1]).
    /// The result of a quantile has the probability axis first: shape <c>[q.Length] + reduced shape</c>. Integers and bool must be cast to float64 by the caller.</summary>
    internal static bool TryOrder(NDArray a, int[] axes, bool keepdims, double[]? q, QuantileMethod method, out NDArray result)
        => TryOrder(a, axes, keepdims, q, method, NanPolicy.Propagate, out result);

    internal static bool TryOrder(NDArray a, int[] axes, bool keepdims, double[]? q, QuantileMethod method, NanPolicy nan, out NDArray result)
    {
        result = null!;
        var scope = current.Value;
        if (scope is { Enabled: false } && nan == NanPolicy.Propagate) return false;
        NDArray? r = a.DType switch
        {
            DType.Float16 => Order<Half>(a, axes, keepdims, q, method, nan),
            DType.Float32 => Order<float>(a, axes, keepdims, q, method, nan),
            DType.Float64 => Order<double>(a, axes, keepdims, q, method, nan),
            _ => null,
        };
        if (r is null) return false;
        scope?.Count();
        result = r;
        return true;
    }

    static NDArray? Order<T>(NDArray a, int[] axes, bool keepdims, double[]? q, QuantileMethod method, NanPolicy nan) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var buffer = Prepare<T>(a, axes, out int offset, out var lane, out _, out _);
        if (q is not null && lane.Count > 0 && lane.Length == 0) return null;   // numpy fails on an empty slice: left to the general engine's error
        var src = Window(buffer, offset, lane);
        var outShape = Red.OutShape(a.Shape, axes, keepdims);
        if (q is null)
        {
            var medians = new T[lane.Count];
            Reductions.Median<T>(src, lane, medians, nan, MedianAverage.Mean);   // numpy: the mean of the two middle values
            return new NDArray(a.DType, medians, outShape);
        }
        var values = new T[q.Length * lane.Count];
        Statistics.Quantile<T>(src, lane, q, values, method, nan);
        return new NDArray(a.DType, values, new[] { q.Length }.Concat(outShape).ToArray());
    }

    // ------------------------------------------------------------------ sort, argsort, searchsorted

    /// <summary>Tries the kernel for <c>sort</c> of a real array along <paramref name="axis"/> (normalised): NaN last, not stable. Bool must be cast to uint8 by the caller.</summary>
    internal static bool TrySort(NDArray a, int axis, out NDArray result)
        => Finish(a.DType switch
        {
            DType.Int8 => SortOf<sbyte>(a, axis),
            DType.UInt8 => SortOf<byte>(a, axis),
            DType.Int16 => SortOf<short>(a, axis),
            DType.UInt16 => SortOf<ushort>(a, axis),
            DType.Int32 => SortOf<int>(a, axis),
            DType.UInt32 => SortOf<uint>(a, axis),
            DType.Int64 => SortOf<long>(a, axis),
            DType.UInt64 => SortOf<ulong>(a, axis),
            DType.Float16 => SortOf<Half>(a, axis),
            DType.Float32 => SortOf<float>(a, axis),
            DType.Float64 => SortOf<double>(a, axis),
            _ => null,
        }, out result, NanPolicy.Propagate);

    /// <summary>Tries the kernel for the stable <c>argsort</c> (int64 result). Bool must be cast to uint8 by the caller.</summary>
    internal static bool TryArgSort(NDArray a, int axis, out NDArray result)
        => Finish(a.DType switch
        {
            DType.Int8 => ArgSortOf<sbyte>(a, axis),
            DType.UInt8 => ArgSortOf<byte>(a, axis),
            DType.Int16 => ArgSortOf<short>(a, axis),
            DType.UInt16 => ArgSortOf<ushort>(a, axis),
            DType.Int32 => ArgSortOf<int>(a, axis),
            DType.UInt32 => ArgSortOf<uint>(a, axis),
            DType.Int64 => ArgSortOf<long>(a, axis),
            DType.UInt64 => ArgSortOf<ulong>(a, axis),
            DType.Float16 => ArgSortOf<Half>(a, axis),
            DType.Float32 => ArgSortOf<float>(a, axis),
            DType.Float64 => ArgSortOf<double>(a, axis),
            _ => null,
        }, out result, NanPolicy.Propagate);

    static NDArray? SortOf<T>(NDArray a, int axis) where T : unmanaged, INumber<T>
    {
        if (!Active) return null;
        var shape = (int[])a.Shape.Clone();
        var values = new T[a.Size];
        if (a.Size > 0)
        {
            var buffer = CContiguous<T>(a, out int offset);
            var lane = Lane.ForRowMajorAxis(shape, axis);
            Sorting.Sort<T>(Window(buffer, offset, lane), lane, values);
        }
        return new NDArray(a.DType, values, shape);
    }

    static NDArray? ArgSortOf<T>(NDArray a, int axis) where T : unmanaged, INumber<T>
    {
        if (!Active) return null;
        var shape = (int[])a.Shape.Clone();
        var order = new int[a.Size];
        if (a.Size > 0)
        {
            var buffer = CContiguous<T>(a, out int offset);
            var lane = Lane.ForRowMajorAxis(shape, axis);
            Sorting.ArgSort<T>(Window(buffer, offset, lane), lane, order);
        }
        var result = new long[order.Length];
        for (int i = 0; i < result.Length; i++) result[i] = order[i];
        return new NDArray(DType.Int64, result, shape);
    }

    /// <summary>The binary search of <paramref name="values"/> in the sorted 1-D <paramref name="sorted"/>, both already cast to one real dtype (not bool). There is no other implementation,
    /// so the on/off switch does not apply. Returns null for a dtype the kernels do not take.</summary>
    internal static NDArray? SearchSorted(NDArray sorted, NDArray values, bool right) => sorted.DType switch
    {
        DType.Int8 => Search<sbyte>(sorted, values, right),
        DType.UInt8 => Search<byte>(sorted, values, right),
        DType.Int16 => Search<short>(sorted, values, right),
        DType.UInt16 => Search<ushort>(sorted, values, right),
        DType.Int32 => Search<int>(sorted, values, right),
        DType.UInt32 => Search<uint>(sorted, values, right),
        DType.Int64 => Search<long>(sorted, values, right),
        DType.UInt64 => Search<ulong>(sorted, values, right),
        DType.Float16 => Search<Half>(sorted, values, right),
        DType.Float32 => Search<float>(sorted, values, right),
        DType.Float64 => Search<double>(sorted, values, right),
        _ => null,
    };

    static NDArray Search<T>(NDArray sorted, NDArray values, bool right) where T : unmanaged, INumber<T>
    {
        var s = CContiguous<T>(sorted, out int so);
        var v = CContiguous<T>(values, out int vo);
        var found = new int[values.Size];
        Sorting.SearchSorted<T>(s.AsSpan(so, sorted.Size), v.AsSpan(vo, values.Size), found, right ? SearchSide.Right : SearchSide.Left);
        var result = new long[found.Length];
        for (int i = 0; i < result.Length; i++) result[i] = found[i];
        CountCall();
        return new NDArray(DType.Int64, result, (int[])values.Shape.Clone());
    }

    // ------------------------------------------------------------------ covariance, correlation

    /// <summary>Tries the kernel for the covariance of the rows of the 2-D float64 array <paramref name="x"/> (one variable per row); the result is variables × variables.</summary>
    internal static bool TryCov(NDArray x, int ddof, out NDArray result)
    {
        result = null!;
        if (!Active || x.DType != DType.Float64 || x.Ndim != 2) return false;
        var buffer = CContiguous<double>(x, out int offset);
        int v = x.Shape[0], n = x.Shape[1];
        var cov = new double[v * v];
        // numpy takes the means along the observations of the array it was given: pairwise only when they are adjacent there (not for the transpose of a C array)
        var meanSum = n <= 1 || x.Strides[1] == 1 ? SumMethod.Pairwise : SumMethod.SequentialFromFirst;
        Products.Covariance<double>(v * n == 0 ? default : buffer.AsSpan(offset, v * n), v, n, cov, ddof, meanSum);
        CountCall();
        result = new NDArray(DType.Float64, cov, new[] { v, v });
        return true;
    }

    /// <summary>Tries the kernel for the correlation matrix from the square float64 covariance matrix <paramref name="cov"/>.</summary>
    internal static bool TryCorrelation(NDArray cov, out NDArray result)
    {
        result = null!;
        if (!Active || cov.DType != DType.Float64 || cov.Ndim != 2 || cov.Shape[0] != cov.Shape[1]) return false;
        var buffer = CContiguous<double>(cov, out int offset);
        int v = cov.Shape[0];
        var corr = new double[v * v];
        Products.Correlation<double>(v == 0 ? default : buffer.AsSpan(offset, v * v), v, corr);
        CountCall();
        result = new NDArray(DType.Float64, corr, new[] { v, v });
        return true;
    }

    // ------------------------------------------------------------------ any, all, count_nonzero, argmax, argmin, diff

    static NDArray? WithT(NDArray a, Func<NDArray, NDArray?> byType) => a.DType == DType.Bool ? byType(a.CastTo(DType.UInt8)) : byType(a);

    /// <summary>Tries the kernels for <c>any</c> (<paramref name="all"/> false) or <c>all</c> over <paramref name="axes"/>. NaN counts as true. The result is bool.</summary>
    internal static bool TryTruth(bool all, NDArray a, int[] axes, bool keepdims, out NDArray result)
        => Finish(WithT(a, x => x.DType switch
        {
            DType.Int8 => Truth<sbyte>(all, x, axes, keepdims),
            DType.UInt8 => Truth<byte>(all, x, axes, keepdims),
            DType.Int16 => Truth<short>(all, x, axes, keepdims),
            DType.UInt16 => Truth<ushort>(all, x, axes, keepdims),
            DType.Int32 => Truth<int>(all, x, axes, keepdims),
            DType.UInt32 => Truth<uint>(all, x, axes, keepdims),
            DType.Int64 => Truth<long>(all, x, axes, keepdims),
            DType.UInt64 => Truth<ulong>(all, x, axes, keepdims),
            DType.Float16 => Truth<Half>(all, x, axes, keepdims),
            DType.Float32 => Truth<float>(all, x, axes, keepdims),
            DType.Float64 => Truth<double>(all, x, axes, keepdims),
            _ => null,
        }), out result, nan: NanPolicy.Propagate);

    /// <summary>Tries the kernels for <c>count_nonzero</c> over <paramref name="axes"/> (int64 result; NaN counts as non-zero).</summary>
    internal static bool TryCountNonzero(NDArray a, int[] axes, bool keepdims, out NDArray result)
        => Finish(WithT(a, x => x.DType switch
        {
            DType.Int8 => Count<sbyte>(x, axes, keepdims),
            DType.UInt8 => Count<byte>(x, axes, keepdims),
            DType.Int16 => Count<short>(x, axes, keepdims),
            DType.UInt16 => Count<ushort>(x, axes, keepdims),
            DType.Int32 => Count<int>(x, axes, keepdims),
            DType.UInt32 => Count<uint>(x, axes, keepdims),
            DType.Int64 => Count<long>(x, axes, keepdims),
            DType.UInt64 => Count<ulong>(x, axes, keepdims),
            DType.Float16 => Count<Half>(x, axes, keepdims),
            DType.Float32 => Count<float>(x, axes, keepdims),
            DType.Float64 => Count<double>(x, axes, keepdims),
            _ => null,
        }), out result, nan: NanPolicy.Propagate);

    /// <summary>Tries the kernels for <c>argmax</c>/<c>argmin</c> (and, with <see cref="NanPolicy.Omit"/>, <c>nanargmax</c>/<c>nanargmin</c>) over <paramref name="axes"/>.
    /// The result is int64. <paramref name="allNan"/> tells that a slice held only NaN (only meaningful with Omit).</summary>
    internal static bool TryArg(bool max, NDArray a, int[] axes, bool keepdims, NanPolicy nan, out NDArray result, out bool allNan)
    {
        bool bad = false;
        bool ok = Finish(WithT(a, x =>
        {
            (NDArray? r, bool b) = x.DType switch
            {
                DType.Int8 => Arg<sbyte>(max, x, axes, keepdims, nan),
                DType.UInt8 => Arg<byte>(max, x, axes, keepdims, nan),
                DType.Int16 => Arg<short>(max, x, axes, keepdims, nan),
                DType.UInt16 => Arg<ushort>(max, x, axes, keepdims, nan),
                DType.Int32 => Arg<int>(max, x, axes, keepdims, nan),
                DType.UInt32 => Arg<uint>(max, x, axes, keepdims, nan),
                DType.Int64 => Arg<long>(max, x, axes, keepdims, nan),
                DType.UInt64 => Arg<ulong>(max, x, axes, keepdims, nan),
                DType.Float16 => Arg<Half>(max, x, axes, keepdims, nan),
                DType.Float32 => Arg<float>(max, x, axes, keepdims, nan),
                DType.Float64 => Arg<double>(max, x, axes, keepdims, nan),
                _ => ((NDArray?)null, false),
            };
            bad = b;
            return r;
        }), out result, nan);
        allNan = bad;
        return ok;
    }

    /// <summary>Tries the kernels for <c>diff</c> of order <paramref name="n"/> along <paramref name="axis"/> (normalised); not for bool, which numpy compares.</summary>
    internal static bool TryDiff(NDArray a, int n, int axis, out NDArray result)
        => Finish(a.DType switch
        {
            DType.Int8 => Difference<sbyte>(a, n, axis),
            DType.UInt8 => Difference<byte>(a, n, axis),
            DType.Int16 => Difference<short>(a, n, axis),
            DType.UInt16 => Difference<ushort>(a, n, axis),
            DType.Int32 => Difference<int>(a, n, axis),
            DType.UInt32 => Difference<uint>(a, n, axis),
            DType.Int64 => Difference<long>(a, n, axis),
            DType.UInt64 => Difference<ulong>(a, n, axis),
            DType.Float16 => Difference<Half>(a, n, axis),
            DType.Float32 => Difference<float>(a, n, axis),
            DType.Float64 => Difference<double>(a, n, axis),
            _ => null,
        }, out result, nan: NanPolicy.Propagate);

    static bool Finish(NDArray? r, out NDArray result, NanPolicy nan)
    {
        result = null!;
        var scope = current.Value;
        if (scope is { Enabled: false } && nan == NanPolicy.Propagate) return false;
        if (r is null) return false;
        scope?.Count();
        result = r;
        return true;
    }

    static NDArray Truth<T>(bool all, NDArray a, int[] axes, bool keepdims) where T : unmanaged, INumberBase<T>
    {
        if (current.Value is { Enabled: false }) return null!;
        var buffer = Prepare<T>(a, axes, out int offset, out var lane, out _, out _);
        var values = new bool[lane.Count];
        var src = Window(buffer, offset, lane);
        if (all) Reductions.All<T>(src, lane, values, true); else Reductions.Any<T>(src, lane, values, true);
        return new NDArray(DType.Bool, values, Red.OutShape(a.Shape, axes, keepdims));
    }

    static NDArray Count<T>(NDArray a, int[] axes, bool keepdims) where T : unmanaged, INumberBase<T>
    {
        if (current.Value is { Enabled: false }) return null!;
        var buffer = Prepare<T>(a, axes, out int offset, out var lane, out _, out _);
        var counts = new int[lane.Count];
        Reductions.CountNonZero<T>(Window(buffer, offset, lane), lane, counts, true);
        var values = new long[counts.Length];
        for (int i = 0; i < values.Length; i++) values[i] = counts[i];
        return new NDArray(DType.Int64, values, Red.OutShape(a.Shape, axes, keepdims));
    }

    static (NDArray?, bool) Arg<T>(bool max, NDArray a, int[] axes, bool keepdims, NanPolicy nan) where T : unmanaged, INumber<T>
    {
        if (nan == NanPolicy.Propagate && current.Value is { Enabled: false }) return (null, false);
        var buffer = Prepare<T>(a, axes, out int offset, out var lane, out _, out _);
        if (lane.Length == 0) throw new NDValueException($"attempt to get {(max ? "argmax" : "argmin")} of an empty sequence");
        var src = Window(buffer, offset, lane);
        var values = new T[lane.Count];
        var indices = new int[lane.Count];
        if (max) Reductions.Max<T>(src, lane, values, indices, nan); else Reductions.Min<T>(src, lane, values, indices, nan);
        bool allNan = false;
        var result = new long[indices.Length];
        for (int i = 0; i < result.Length; i++) { result[i] = indices[i]; if (T.IsNaN(values[i])) allNan = true; }
        // with NaN propagated a NaN value is a legitimate answer (the index of the first NaN); only the omitting variant reports an all-NaN slice
        return (new NDArray(DType.Int64, result, Red.OutShape(a.Shape, axes, keepdims)), nan == NanPolicy.Omit && allNan);
    }

    static NDArray Difference<T>(NDArray a, int n, int axis) where T : unmanaged, INumber<T>
    {
        if (current.Value is { Enabled: false }) return null!;
        var shape = (int[])a.Shape.Clone();
        var buffer = CContiguous<T>(a, out int offset);
        var lane = Lane.ForRowMajorAxis(shape, axis);
        int outLength = Math.Max(shape[axis] - n, 0);
        var outShape = (int[])shape.Clone();
        outShape[axis] = outLength;
        var outLane = new Lane(outLength, lane.Inner, lane.Outer);
        var values = new T[outLane.Total];
        Scans.Difference<T>(Window(buffer, offset, lane), lane, values, n);
        return new NDArray(a.DType, values, outShape);
    }

    // ------------------------------------------------------------------ gradient, trapezoid, histogram

    /// <summary>The elements of <paramref name="a"/> in C order, in place when it is already C-contiguous.</summary>
    static T[] CContiguous<T>(NDArray a, out int offset) where T : unmanaged
    {
        var shape = (int[])a.Shape.Clone();
        var strides = a.Strides;
        var buffer = (T[])a.Buffer;
        if (IsCContiguous(shape, strides)) { offset = a.Offset; return buffer; }
        offset = 0;
        int size = a.Size;
        var copy = new T[size];
        var idx = new int[shape.Length];
        int pos = a.Offset;
        for (int k = 0; k < size; k++)
        {
            copy[k] = buffer[pos];
            for (int d = shape.Length - 1; d >= 0; d--)
            {
                idx[d]++; pos += strides[d];
                if (idx[d] < shape[d]) break;
                pos -= strides[d] * shape[d]; idx[d] = 0;
            }
        }
        return copy;
    }

    /// <summary>Tries the kernels for <c>gradient</c> along <paramref name="axis"/> (normalised, at least two elements long) of a float32 or float64 array:
    /// uniform spacing <paramref name="h"/>, or the sample positions <paramref name="x"/> (same dtype as <paramref name="f"/>, strictly one value per element).
    /// The result is C-contiguous with the shape of <paramref name="f"/>.</summary>
    internal static bool TryGradient(NDArray f, int axis, double h, NDArray? x, out NDArray result)
    {
        result = null!;
        var scope = current.Value;
        if (scope is { Enabled: false }) return false;
        NDArray? r = f.DType switch
        {
            DType.Float16 => Gradient<Half>(f, axis, h, x),
            DType.Float32 => Gradient<float>(f, axis, h, x),
            DType.Float64 => Gradient<double>(f, axis, h, x),
            _ => null,
        };
        if (r is null) return false;
        scope?.Count();
        result = r;
        return true;
    }

    static NDArray? Gradient<T>(NDArray f, int axis, double h, NDArray? x) where T : unmanaged, IFloatingPointIeee754<T>
    {
        if (x is not null && x.DType != f.DType) return null;   // mixed dtypes: numpy promotes; left to the general engine
        var shape = (int[])f.Shape.Clone();
        var buffer = CContiguous<T>(f, out int offset);
        var lane = Lane.ForRowMajorAxis(shape, axis);
        var values = new T[f.Size];
        var src = Window(buffer, offset, lane);
        if (x is null) Statistics.Gradient<T>(src, lane, T.CreateChecked(h), values);
        else
        {
            var xs = CContiguous<T>(x, out int xo);
            Statistics.Gradient<T>(src, lane, xs.AsSpan(xo, x.Size), values);
        }
        return new NDArray(f.DType, values, shape);
    }

    /// <summary>Tries the kernels for the trapezoidal rule along <paramref name="axis"/> (normalised) of a float32 or float64 array, with spacing
    /// <paramref name="dx"/> or the positions <paramref name="x"/> (1-D, same dtype). The terms are added as NumPy adds them: pairwise only when the axis ends the shape.</summary>
    internal static bool TryTrapezoid(NDArray y, int axis, double dx, NDArray? x, out NDArray result)
    {
        result = null!;
        var scope = current.Value;
        if (scope is { Enabled: false }) return false;
        NDArray? r = y.DType switch
        {
            DType.Float16 => Trapezoid<Half>(y, axis, dx, x),
            DType.Float32 => Trapezoid<float>(y, axis, dx, x),
            DType.Float64 => Trapezoid<double>(y, axis, dx, x),
            _ => null,
        };
        if (r is null) return false;
        scope?.Count();
        result = r;
        return true;
    }

    static NDArray? Trapezoid<T>(NDArray y, int axis, double dx, NDArray? x) where T : unmanaged, IFloatingPointIeee754<T>
    {
        if (x is not null && x.DType != y.DType) return null;
        var shape = (int[])y.Shape.Clone();
        var buffer = CContiguous<T>(y, out int offset);
        var lane = Lane.ForRowMajorAxis(shape, axis);
        // NumPy forms the terms as a new C-contiguous array and sums it along the axis: pairwise only if the axis is the last non-trivial one.
        var method = Product(shape, axis + 1, shape.Length) == 1 ? SumMethod.Pairwise : SumMethod.SequentialFromFirst;
        var values = new T[lane.Count];
        var src = Window(buffer, offset, lane);
        if (x is null) Statistics.Trapezoid<T>(src, lane, T.CreateChecked(dx), values, method);
        else
        {
            var xs = CContiguous<T>(x, out int xo);
            Statistics.Trapezoid<T>(src, lane, xs.AsSpan(xo, x.Size), values, method);
        }
        return new NDArray(y.DType, values, Red.OutShape(shape, new[] { axis }, false));
    }

    /// <summary>Counts the finite-edge histogram of the elements of <paramref name="data"/> (any real dtype, as float64) in the bins given by <paramref name="edges"/>
    /// (non-decreasing); the last bin is closed, as NumPy's. Values outside the edges and NaN are not counted.</summary>
    internal static long[] Histogram(NDArray data, double[] edges)
    {
        var d = data.DType == DType.Float64 ? data : data.CastTo(DType.Float64);
        var buffer = CContiguous<double>(d, out int offset);
        var counts = new long[Math.Max(edges.Length - 1, 0)];
        if (counts.Length > 0)
            Statistics.Histogram<double>(buffer.AsSpan(offset, d.Size), Lane.Flat(d.Size), edges, counts, HistogramLastBin.ClosedInterval);
        scopeCount();
        return counts;
    }

    static void scopeCount() => current.Value?.Count();

    // ------------------------------------------------------------------ scans

    static NDArray Scan<T>(Op op, NDArray a, int? axis, NanPolicy nan) where T : unmanaged, INumber<T>
    {
        var shape = (int[])a.Shape.Clone();
        var strides = a.Strides;
        var buffer = (T[])a.Buffer;
        int size = a.Size;
        int[] resultShape = axis is null ? [size] : shape;
        var values = new T[size];
        if (size > 0)
        {
            T[] contiguous; int offset;
            if (IsCContiguous(shape, strides)) { contiguous = buffer; offset = a.Offset; }
            else
            {
                contiguous = new T[size]; offset = 0;
                var idx = new int[shape.Length];
                int pos = a.Offset;
                for (int k = 0; k < size; k++)
                {
                    contiguous[k] = buffer[pos];
                    for (int d = shape.Length - 1; d >= 0; d--)
                    {
                        idx[d]++; pos += strides[d];
                        if (idx[d] < shape[d]) break;
                        pos -= strides[d] * shape[d]; idx[d] = 0;
                    }
                }
            }
            var lane = axis is null ? Lane.Flat(size) : Lane.ForRowMajorAxis(shape, axis.Value);
            Scans.Scan<T>(contiguous.AsSpan(offset, size), lane, values, op == Op.CumSum ? ScanOp.Sum : ScanOp.Product, ScanDirection.Forward, nan);
        }
        return new NDArray(a.DType, values, resultShape);
    }
}
