// Copyright (c) 2026 Marco Parenzan
//
// Licensed under the MIT License. See the LICENSE file in the project
// root for full license information.

using System.Numerics;
using MathemaSharp.Kernels;

namespace NDSharp;

/// <summary>Runs <c>sum prod max min mean var std cumsum cumprod</c> on the lane kernels of <c>MathemaSharp.Kernels</c>, for every real dtype
/// except float16 (integers, float32, float64), over any axes, any layout and empty arrays, with numpy's rules: NaN propagates, sums are
/// pairwise when the reduced elements are adjacent in memory (and plain from the first element otherwise), variance is two-pass.
/// A C-contiguous array whose reduced axes are adjacent is read in place; any other layout is first gathered into a contiguous copy with the
/// reduced axes last. Float16, complex and the temporal dtypes are left to the general engine.</summary>
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

    internal enum Op { Sum, Prod, Max, Min, Mean, Var, CumSum, CumProd }

    /// <summary>Tries the kernels. <paramref name="a"/> is already cast to the dtype of the computation; <paramref name="axes"/> is the normalised,
    /// sorted list of reduced axes (all of them when the caller passed none).</summary>
    internal static bool TryReduce(Op op, NDArray a, int[] axes, bool keepdims, int ddof, out NDArray result)
    {
        result = null!;
        var scope = current.Value;
        if (scope is { Enabled: false }) return false;
        NDArray? r = op is Op.Mean or Op.Var ? ReduceFloat(op, a, axes, keepdims, ddof) : ReduceNumber(op, a, axes, keepdims);
        if (r is null) return false;
        scope?.Count();
        result = r;
        return true;
    }

    /// <summary>Tries the kernels for <c>cumsum</c> and <c>cumprod</c>; <paramref name="a"/> is already cast; <paramref name="axis"/> is normalised, or null for the flattened array.</summary>
    internal static bool TryScan(Op op, NDArray a, int? axis, out NDArray result)
    {
        result = null!;
        var scope = current.Value;
        if (scope is { Enabled: false }) return false;
        NDArray? r = a.DType switch
        {
            DType.Int8 => Scan<sbyte>(op, a, axis),
            DType.UInt8 => Scan<byte>(op, a, axis),
            DType.Int16 => Scan<short>(op, a, axis),
            DType.UInt16 => Scan<ushort>(op, a, axis),
            DType.Int32 => Scan<int>(op, a, axis),
            DType.UInt32 => Scan<uint>(op, a, axis),
            DType.Int64 => Scan<long>(op, a, axis),
            DType.UInt64 => Scan<ulong>(op, a, axis),
            DType.Float32 => Scan<float>(op, a, axis),
            DType.Float64 => Scan<double>(op, a, axis),
            _ => null,
        };
        if (r is null) return false;
        scope?.Count();
        result = r;
        return true;
    }

    static NDArray? ReduceNumber(Op op, NDArray a, int[] axes, bool keepdims) => a.DType switch
    {
        DType.Int8 => Reduce<sbyte>(op, a, axes, keepdims),
        DType.UInt8 => Reduce<byte>(op, a, axes, keepdims),
        DType.Int16 => Reduce<short>(op, a, axes, keepdims),
        DType.UInt16 => Reduce<ushort>(op, a, axes, keepdims),
        DType.Int32 => Reduce<int>(op, a, axes, keepdims),
        DType.UInt32 => Reduce<uint>(op, a, axes, keepdims),
        DType.Int64 => Reduce<long>(op, a, axes, keepdims),
        DType.UInt64 => Reduce<ulong>(op, a, axes, keepdims),
        DType.Float32 => Reduce<float>(op, a, axes, keepdims),
        DType.Float64 => Reduce<double>(op, a, axes, keepdims),
        _ => null,
    };

    static NDArray? ReduceFloat(Op op, NDArray a, int[] axes, bool keepdims, int ddof) => a.DType switch
    {
        DType.Float32 => MeanOrVar<float>(op, a, axes, keepdims, ddof),
        DType.Float64 => MeanOrVar<double>(op, a, axes, keepdims, ddof),
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

    static NDArray Reduce<T>(Op op, NDArray a, int[] axes, bool keepdims) where T : unmanaged, INumber<T>
    {
        var buffer = Prepare<T>(a, axes, out int offset, out var lane, out var method, out _);
        var src = Window(buffer, offset, lane);
        var values = new T[lane.Count];
        switch (op)
        {
            case Op.Sum: Reductions.Sum<T>(src, lane, values, NanPolicy.Propagate, method); break;
            case Op.Prod: Reductions.Product<T>(src, lane, values); break;
            case Op.Max:
            case Op.Min:
                if (lane.Count > 0 && lane.Length == 0)
                    throw new NDValueException($"zero-size array to reduction operation {(op == Op.Max ? "maximum" : "minimum")} which has no identity");
                if (op == Op.Max) Reductions.Max<T>(src, lane, values, Span<int>.Empty, NanPolicy.Propagate);
                else Reductions.Min<T>(src, lane, values, Span<int>.Empty, NanPolicy.Propagate);
                break;
            default: throw new InvalidOperationException(op.ToString());
        }
        return new NDArray(a.DType, values, Red.OutShape(a.Shape, axes, keepdims));
    }

    static NDArray MeanOrVar<T>(Op op, NDArray a, int[] axes, bool keepdims, int ddof) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var buffer = Prepare<T>(a, axes, out int offset, out var lane, out var method, out var deviationMethod);
        var src = Window(buffer, offset, lane);
        var values = new T[lane.Count];
        if (op == Op.Mean) Reductions.Mean<T>(src, lane, values, NanPolicy.Propagate, method);
        else
        {
            // numpy divides by max(n - ddof, 0): a positive sum of squares over zero gives +inf, zero or NaN gives NaN (the kernel gives NaN for n - ddof <= 0).
            bool degenerate = lane.Length - ddof <= 0;
            Reductions.Variance<T>(src, lane, values, degenerate ? 0 : Math.Max(ddof, 0), NanPolicy.Propagate, VarianceMethod.TwoPass, method, deviationMethod);
            if (degenerate)
                for (int i = 0; i < values.Length; i++) values[i] = values[i] > T.Zero ? T.PositiveInfinity : T.NaN;
        }
        return new NDArray(a.DType, values, Red.OutShape(a.Shape, axes, keepdims));
    }

    // ------------------------------------------------------------------ order statistics

    /// <summary>Tries the kernels for <c>median</c> (<paramref name="q"/> null) or the quantiles <paramref name="q"/> (probabilities in [0, 1]).
    /// The result of a quantile has the probability axis first: shape <c>[q.Length] + reduced shape</c>. Integers and bool must be cast to float64 by the caller.</summary>
    internal static bool TryOrder(NDArray a, int[] axes, bool keepdims, double[]? q, QuantileMethod method, out NDArray result)
    {
        result = null!;
        var scope = current.Value;
        if (scope is { Enabled: false }) return false;
        NDArray? r = a.DType switch
        {
            DType.Float32 => Order<float>(a, axes, keepdims, q, method),
            DType.Float64 => Order<double>(a, axes, keepdims, q, method),
            _ => null,
        };
        if (r is null) return false;
        scope?.Count();
        result = r;
        return true;
    }

    static NDArray? Order<T>(NDArray a, int[] axes, bool keepdims, double[]? q, QuantileMethod method) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var buffer = Prepare<T>(a, axes, out int offset, out var lane, out _, out _);
        if (q is not null && lane.Count > 0 && lane.Length == 0) return null;   // numpy fails on an empty slice: left to the general engine's error
        var src = Window(buffer, offset, lane);
        var outShape = Red.OutShape(a.Shape, axes, keepdims);
        if (q is null)
        {
            var medians = new T[lane.Count];
            Reductions.Median<T>(src, lane, medians, NanPolicy.Propagate, MedianAverage.Mean);   // numpy: the mean of the two middle values
            return new NDArray(a.DType, medians, outShape);
        }
        var values = new T[q.Length * lane.Count];
        Statistics.Quantile<T>(src, lane, q, values, method, NanPolicy.Propagate);
        return new NDArray(a.DType, values, new[] { q.Length }.Concat(outShape).ToArray());
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

    static NDArray Scan<T>(Op op, NDArray a, int? axis) where T : unmanaged, INumber<T>
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
            Scans.Scan<T>(contiguous.AsSpan(offset, size), lane, values, op == Op.CumSum ? ScanOp.Sum : ScanOp.Product, ScanDirection.Forward, NanPolicy.Propagate);
        }
        return new NDArray(a.DType, values, resultShape);
    }
}
