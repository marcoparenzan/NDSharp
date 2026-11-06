// Copyright (c) 2026 Marco Parenzan
//
// Licensed under the MIT License. See the LICENSE file in the project
// root for full license information.

using System.Numerics;
using MathemaSharp.Kernels;

namespace NDSharp;

public static partial class np
{
    // ================================================================ sorting

    /// <summary>numpy <c>sort</c> along <paramref name="axis"/> (default last); <c>null</c> sorts the flattened array. Runs on the MathemaSharp sorting kernels: NaN last,
    /// complex numbers in numpy's lexicographic order, not stable (as numpy's default sort).</summary>
    public static NDArray Sort(NDArray a, int? axis = -1)
    {
        if (axis is null) return Sort(Flatten(a), 0);
        if (a.Ndim == 0) throw new NDAxisException(axis.Value, 0);
        int ax = NormalizeAxis(axis.Value, a.Ndim);
        if (a.DType == DType.Bool) return Sort(a.AsType(DType.UInt8), ax).AsType(DType.Bool);
        return KernelPath.TrySort(a, ax, out var sorted) ? sorted : throw new NDNotSupportedException($"sort is not implemented for {a.DType}");
    }

    /// <summary>numpy <c>argsort</c> (stable: equal values keep their index order; NaN last; complex numbers in numpy's lexicographic order). Runs on the MathemaSharp sorting kernels.</summary>
    public static NDArray ArgSort(NDArray a, int? axis = -1)
    {
        if (axis is null) return ArgSort(Flatten(a), 0);
        if (a.Ndim == 0) throw new NDAxisException(axis.Value, 0);
        int ax = NormalizeAxis(axis.Value, a.Ndim);
        var src = a.DType == DType.Bool ? a.AsType(DType.UInt8) : a;
        return KernelPath.TryArgSort(src, ax, out var order) ? order : throw new NDNotSupportedException($"argsort is not implemented for {a.DType}");
    }

    /// <summary>
    /// numpy <c>sort_complex</c>: sorts along the last axis by real part, then imaginary part, and returns a complex array
    /// (complex128, except complex64 for 8/16-bit integer input and for complex64 itself).
    /// </summary>
    public static NDArray SortComplex(NDArray a)
    {
        var sorted = Sort(a, -1);
        if (sorted.DType.IsComplex()) return sorted;
        var target = sorted.DType is DType.Int8 or DType.UInt8 or DType.Int16 or DType.UInt16 ? DType.Complex64 : DType.Complex128;
        return sorted.AsType(target);
    }

    // ================================================================ sets & counting

    public sealed record UniqueResult(NDArray Values, NDArray? Index, NDArray? Inverse, NDArray? Counts);

    /// <summary>numpy <c>unique</c> of the flattened array (sorted).</summary>
    public static UniqueResult Unique(NDArray a, bool returnIndex = false, bool returnInverse = false, bool returnCounts = false)
    {
        var flat = Flatten(a);
        int n = flat.Size;
        var order = ArgSort(flat, 0).ToArray<long>();
        var sortedVals = flat.Get(NDArray.FromArray(order)).Copy();
        var keep = new List<int>();
        var counts = new List<long>();
        var inverse = new long[n];
        var eq = n > 1 ? np.Equal(sortedVals.Get(new Slice(1, null)), sortedVals.Get(new Slice(null, -1))).ToArray<bool>() : Array.Empty<bool>();
        // NaNs: numpy >= 1.21 collapses all NaNs into one entry.
        var isNan = sortedVals.DType.IsFloat() || sortedVals.DType.IsComplex() ? np.IsNan(sortedVals).ToArray<bool>() : new bool[n];
        int u = -1;
        for (int i = 0; i < n; i++)
        {
            bool newGroup = i == 0 || !(eq[i - 1] || (isNan[i] && isNan[i - 1]));
            if (newGroup)
            {
                keep.Add(i);
                counts.Add(0);
                u++;
            }
            counts[u]++;
            inverse[order[i]] = u;
        }
        var values = sortedVals.Get(NDArray.FromArray(keep.Select(i => (long)i).ToArray())).Copy();
        if (keep.Count == 0) values = np.Zeros(new[] { 0 }, a.DType);
        return new UniqueResult(
            values,
            returnIndex ? NDArray.FromArray(keep.Select(i => order[i]).ToArray()) : null,
            returnInverse ? NDArray.FromArray(inverse, a.Shape.Length == 0 ? new[] { 1 } : new[] { n }) : null,
            returnCounts ? NDArray.FromArray(counts.ToArray()) : null);
    }

    /// <summary>numpy <c>bincount</c>.</summary>
    public static NDArray BinCount(NDArray x, NDArray? weights = null, int minlength = 0)
    {
        if (x.Ndim != 1) throw new NDValueException("object too deep for desired array");
        if (!x.DType.IsInteger() && x.DType != DType.Bool)
            throw new NDTypeException("Cannot cast array data from dtype('float64') to dtype('int64') according to the rule 'safe'");
        var vals = x.ToArray<long>();
        if (vals.Any(v => v < 0)) throw new NDValueException("'list' argument must have no negative elements");
        int len = Math.Max(vals.Length == 0 ? 0 : (int)vals.Max() + 1, minlength);
        if (weights is null)
        {
            var counts = new long[len];
            foreach (var v in vals) counts[v]++;
            return NDArray.FromArray(counts, len);
        }
        var w = weights.ToArray<double>();
        var sums = new double[len];
        for (int i = 0; i < vals.Length; i++) sums[vals[i]] += w[i];
        return NDArray.FromArray(sums, len);
    }

    /// <summary>numpy <c>unravel_index</c> (C order); outputs have the shape of <paramref name="indices"/>.</summary>
    public static NDArray[] UnravelIndex(NDArray indices, int[] shape)
    {
        var idx = indices.ToArray<long>();
        long total = 1;
        foreach (var d in shape) total *= d;
        var outs = shape.Select(_ => new long[idx.Length]).ToArray();
        for (int i = 0; i < idx.Length; i++)
        {
            long v = idx[i];
            if (v < 0 || v >= total) throw new NDValueException("index " + v + " is out of bounds for array with size " + total);
            for (int d = shape.Length - 1; d >= 0; d--)
            {
                outs[d][i] = v % shape[d];
                v /= shape[d];
            }
        }
        return outs.Select(o => new NDArray(DType.Int64, o, (int[])indices.Shape.Clone())).ToArray();
    }

    // ================================================================ order statistics

    /// <summary>numpy <c>median</c>; NaN if the slice contains a NaN.</summary>
    public static NDArray Median(NDArray a, int[]? axis = null, bool keepdims = false)
    {
        var axes = Red.NormalizeAxes(axis, a.Ndim);
        if (KernelPath.TryOrder(ForOrder(a), axes, keepdims, null, default, out var fast)) return fast;
        throw new NDNotSupportedException($"median is not implemented for {a.DType}");
    }

    /// <summary>Integers and bool are ordered as float64 (what numpy's median and percentile return for them); other dtypes are untouched.</summary>
    private static NDArray ForOrder(NDArray a) => a.DType.Kind() is 'b' or 'i' or 'u' ? a.CastTo(DType.Float64) : a;

    /// <summary>numpy's <c>method=</c> names, as the kernels' <see cref="MathemaSharp.Kernels.QuantileMethod"/>.</summary>
    private static MathemaSharp.Kernels.QuantileMethod QuantileMethodOf(string method) => method switch
    {
        "linear" => MathemaSharp.Kernels.QuantileMethod.Linear,
        "inverted_cdf" => MathemaSharp.Kernels.QuantileMethod.InvertedCdf,
        "averaged_inverted_cdf" => MathemaSharp.Kernels.QuantileMethod.AveragedInvertedCdf,
        "interpolated_inverted_cdf" => MathemaSharp.Kernels.QuantileMethod.InterpolatedInvertedCdf,
        "hazen" => MathemaSharp.Kernels.QuantileMethod.Hazen,
        "weibull" => MathemaSharp.Kernels.QuantileMethod.Weibull,
        "median_unbiased" => MathemaSharp.Kernels.QuantileMethod.MedianUnbiased,
        "normal_unbiased" => MathemaSharp.Kernels.QuantileMethod.NormalUnbiased,
        "lower" => MathemaSharp.Kernels.QuantileMethod.Lower,
        "higher" => MathemaSharp.Kernels.QuantileMethod.Higher,
        "midpoint" => MathemaSharp.Kernels.QuantileMethod.Midpoint,
        "nearest" => MathemaSharp.Kernels.QuantileMethod.Nearest,
        "closest_observation" => MathemaSharp.Kernels.QuantileMethod.ClosestObservation,
        _ => throw new NDValueException($"{method} is not a valid method. Use one of: inverted_cdf, averaged_inverted_cdf, closest_observation, interpolated_inverted_cdf, hazen, weibull, linear, median_unbiased, normal_unbiased, lower, higher, midpoint, nearest"),
    };

    /// <summary>numpy <c>percentile</c>/<c>quantile</c> (method 'linear'); <paramref name="q"/> in [0, 100] (or [0, 1] when <paramref name="isQuantile"/>).</summary>
    public static NDArray Percentile(NDArray a, NDArray q, int[]? axis = null, bool keepdims = false, bool isQuantile = false)
        => Percentile(a, q, axis, keepdims, isQuantile, "linear");

    /// <summary>numpy <c>percentile</c>/<c>quantile</c> with a <c>method</c>: <c>linear</c> (numpy's default), <c>inverted_cdf</c>, <c>averaged_inverted_cdf</c>,
    /// <c>interpolated_inverted_cdf</c>, <c>hazen</c>, <c>weibull</c>, <c>median_unbiased</c>, <c>normal_unbiased</c>, <c>lower</c>, <c>higher</c>, <c>midpoint</c>, <c>nearest</c>
    /// (all of numpy's). Every method needs float32 or float64 data, or integers and bool, which are ordered as float64.</summary>
    public static NDArray Percentile(NDArray a, NDArray q, int[]? axis, bool keepdims, bool isQuantile, string method)
        => PercentileCore(a, q, axis, keepdims, isQuantile, method, NanPolicy.Propagate);

    private static NDArray PercentileCore(NDArray a, NDArray q, int[]? axis, bool keepdims, bool isQuantile, string method, NanPolicy nan)
    {
        var quantileMethod = QuantileMethodOf(method);
        var axes = Red.NormalizeAxes(axis, a.Ndim);
        var qs = q.AsType(DType.Float64);
        var qv = qs.ToArray<double>().Select(v => isQuantile ? v : v / 100.0).ToArray();
        if (qv.Any(v => v < 0 || v > 1)) throw new NDValueException(isQuantile ? "Quantiles must be in the range [0, 1]" : "Percentiles must be in the range [0, 100]");
        if (KernelPath.TryOrder(ForOrder(a), axes, keepdims, qv, quantileMethod, nan, out var fast))
        {
            var shape = Red.OutShape(a.Shape, axes, keepdims);
            return Reshape(fast, q.Ndim == 0 ? shape : q.Shape.Concat(shape).ToArray());
        }
        if (nan == NanPolicy.Omit) throw new NDValueException("cannot compute a quantile of an empty slice");
        // the kernels leave out a slice with no element (numpy's percentile fails there: IndexError on position -1)
        long reduced = 1;
        foreach (var ax in axes) reduced *= a.Shape[ax];
        if (reduced == 0) throw new NDIndexException("index -1 is out of bounds for axis 0 with size 0");
        throw new NDNotSupportedException($"percentile is not implemented for {a.DType}");
    }

    // ================================================================ differences

    /// <summary>numpy <c>diff</c> along <paramref name="axis"/>, <paramref name="n"/> times.</summary>
    public static NDArray Diff(NDArray a, int n = 1, int axis = -1)
    {
        if (a.Ndim == 0) throw new NDValueException("diff requires input that is at least one dimensional");
        int ax = NormalizeAxis(axis, a.Ndim);
        if (n > 0 && KernelPath.TryDiff(a, n, ax, out var fast)) return fast;
        var r = a;
        for (int k = 0; k < n; k++)
        {
            var idxHi = Enumerable.Repeat<NDIndex>(Slice.All, a.Ndim).ToArray();
            var idxLo = Enumerable.Repeat<NDIndex>(Slice.All, a.Ndim).ToArray();
            idxHi[ax] = new Slice(1, null);
            idxLo[ax] = new Slice(null, -1);
            var hi = r.Get(idxHi);
            var lo = r.Get(idxLo);
            r = r.DType == DType.Bool ? NotEqual(hi, lo) : Subtract(hi, lo);
        }
        return r;
    }

    /// <summary>numpy <c>gradient</c> with uniform spacing (scalar per axis); central differences inside, one-sided at the edges.</summary>
    public static NDArray[] Gradient(NDArray f, double[]? spacing = null, int[]? axes = null)
    {
        var ax = axes is null ? Enumerable.Range(0, f.Ndim).ToArray() : axes.Select(x => NormalizeAxis(x, f.Ndim)).ToArray();
        var ft = f.DType.IsFloat() ? f : f.AsType(DType.Float64);
        var result = new List<NDArray>();
        for (int k = 0; k < ax.Length; k++)
        {
            double dx = spacing is null ? 1.0 : spacing.Length == 1 ? spacing[0] : spacing[k];
            int a0 = ax[k];
            int n = f.Shape[a0];
            if (n < 2) throw new NDValueException("Shape of array too small to calculate a numerical gradient, at least (edge_order + 1) elements are required.");
            if (!KernelPath.TryGradient(ft, a0, dx, null, out var kernelGradient)) throw new NDNotSupportedException($"gradient is not implemented for {ft.DType}");
            result.Add(kernelGradient);
        }
        return result.ToArray();
    }

    /// <summary>numpy <c>gradient</c> with a coordinate array per axis (non-uniform spacing uses numpy's
    /// second-order interior formula; evenly spaced coordinates collapse to the uniform case).</summary>
    public static NDArray[] GradientCoordinates(NDArray f, IReadOnlyList<NDArray> coordinates, int[]? axes = null)
    {
        var ax = axes is null ? Enumerable.Range(0, f.Ndim).ToArray() : axes.Select(x => NormalizeAxis(x, f.Ndim)).ToArray();
        if (coordinates.Count != ax.Length)
            throw new NDValueException("spacing must either be a single scalar, or a scalar / 1d-array per axis");
        var ft = f.DType.IsFloat() ? f : f.AsType(DType.Float64);
        var result = new List<NDArray>();
        for (int k = 0; k < ax.Length; k++)
        {
            var x = coordinates[k];
            int a0 = ax[k], n = f.Shape[a0];
            if (x.Ndim != 1 || x.Size != n)
                throw new NDValueException("when 1d, distances must match the length of the corresponding dimension");
            var dxs = Diff(x.DType.IsFloat() ? x : x.AsType(DType.Float64));
            bool uniform = (bool)All(Equal(dxs, dxs.Get(0))).GetAt(0);
            if (uniform)
            {
                result.Add(Gradient(f, new[] { Convert.ToDouble(dxs.Get(0).AsType(DType.Float64).GetAt(0)) }, new[] { a0 })[0]);
                continue;
            }
            // the positions and the data are processed in their common floating dtype and the result is returned in the data's dtype
            var xf = x.DType.IsFloat() ? x : x.AsType(DType.Float64);
            var common = DTypes.Promote(ft.DType, xf.DType);
            if (!KernelPath.TryGradient(ft.CastTo(common), a0, 0, xf.CastTo(common), out var kernelGradient)) throw new NDNotSupportedException($"gradient is not implemented for {common}");
            result.Add(common == ft.DType ? kernelGradient : kernelGradient.CastTo(ft.DType));
        }
        return result.ToArray();
    }

    // ================================================================ vector algebra

    /// <summary>numpy <c>cross</c> of 3-vectors (or 2-vectors, giving the z component) along the last axis.</summary>
    public static NDArray Cross(NDArray a, NDArray b)
    {
        int la = a.Shape[^1], lb = b.Shape[^1];
        if ((la != 2 && la != 3) || (lb != 2 && lb != 3))
            throw new NDValueException("incompatible dimensions for cross product\n(dimension must be 2 or 3)");
        var rt = DTypes.Promote(a.DType, b.DType);
        NDArray C(NDArray x, int i) => x.Get(NDIndex.Ellipsis, i);
        var zero = NDArray.Scalar(0L, rt);
        NDArray A(int i) => i < la ? C(a, i) : zero;
        NDArray B(int i) => i < lb ? C(b, i) : zero;
        if (la == 2 && lb == 2)
            return Subtract(Multiply(A(0), B(1)), Multiply(A(1), B(0)));
        var x0 = Subtract(Multiply(A(1), B(2)), Multiply(A(2), B(1)));
        var x1 = Subtract(Multiply(A(2), B(0)), Multiply(A(0), B(2)));
        var x2 = Subtract(Multiply(A(0), B(1)), Multiply(A(1), B(0)));
        return Stack(new[] { x0, x1, x2 }, -1);
    }

    /// <summary>numpy <c>convolve</c> (1-D); <paramref name="mode"/> is "full", "same" or "valid".</summary>
    public static NDArray Convolve(NDArray a, NDArray v, string mode = "full")
    {
        if (a.Ndim != 1 || v.Ndim != 1) throw new NDValueException("object too deep for desired array");
        if (a.Size == 0) throw new NDValueException("v cannot be empty");
        if (v.Size == 0) throw new NDValueException("v cannot be empty");
        var rt = DTypes.Promote(a.DType, v.DType);
        if (rt == DType.Bool) rt = DType.Int8;
        int n = a.Size, m = v.Size;
        // Build the full convolution as a sum of shifted, scaled copies (exact and dtype-preserving).
        var full = np.Zeros(new[] { n + m - 1 }, rt);
        var ac = a.CastTo(rt);
        for (int j = 0; j < m; j++)
        {
            var scaled = Multiply(ac, v.CastTo(rt).Get(j));
            var target = full.Get(new Slice(j, j + n));
            Assign.Copy(target, Add(target, scaled), Casting.Unsafe);
        }
        return mode switch
        {
            "full" => full,
            "same" => full.Get(new Slice((Math.Min(n, m) - 1) / 2, (Math.Min(n, m) - 1) / 2 + Math.Max(n, m))).Copy(),
            "valid" => full.Get(new Slice(Math.Min(n, m) - 1, Math.Max(n, m))).Copy(),
            _ => throw new NDValueException("mode must be one of 'valid', 'same', or 'full'"),
        };
    }

    // ================================================================ covariance

    /// <summary>numpy <c>cov</c> (rows are variables unless <paramref name="rowvar"/> is false).</summary>
    public static NDArray Cov(NDArray m, NDArray? y = null, bool rowvar = true, bool bias = false, int? ddof = null)
    {
        int dd = ddof ?? (bias ? 0 : 1);
        var x = AtLeast2d(m.AsType(DTypes.Promote(m.DType, DType.Float64)));
        if (!rowvar && x.Shape[0] != 1) x = Transpose(x);
        if (y is not null)
        {
            var yy = AtLeast2d(y.AsType(DType.Float64));
            if (!rowvar && yy.Shape[0] != 1) yy = Transpose(yy);
            x = VStack(new[] { x, yy });
        }
        int n = x.Shape[1];
        if (x.DType == DType.Float64 && KernelPath.TryCov(x, dd, out var fastCov)) return Squeeze(fastCov);
        throw new NDNotSupportedException($"cov is not implemented for {x.DType}");
    }

    /// <summary>numpy <c>corrcoef</c>.</summary>
    public static NDArray CorrCoef(NDArray x, NDArray? y = null, bool rowvar = true)
    {
        var c = Cov(x, y, rowvar);
        if (c.Ndim == 0) return Divide(c, c);
        return KernelPath.TryCorrelation(c, out var corr) ? corr : throw new NDNotSupportedException($"corrcoef is not implemented for {c.DType}");
    }
}

