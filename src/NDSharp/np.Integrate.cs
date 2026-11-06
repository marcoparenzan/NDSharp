// Copyright (c) 2026 Marco Parenzan
//
// Licensed under the MIT License. See the LICENSE file in the project
// root for full license information.

namespace NDSharp;

public static partial class np
{
    // ================================================================ trapezoid / histogram (on MathemaSharp.Kernels)

    /// <summary>numpy <c>trapezoid</c> (<c>trapz</c>): the integral of <paramref name="y"/> along <paramref name="axis"/> by the trapezoidal rule,
    /// with spacing <paramref name="dx"/> or the sample positions <paramref name="x"/> (1-D with one value per element of the axis, or as many as <paramref name="y"/>).
    /// The terms <c>d·(y[k+1] + y[k])/2</c> are formed first and then added like <c>sum</c>. float32 and float64 run on the MathemaSharp kernels; other dtypes are computed as float64.</summary>
    public static NDArray Trapezoid(NDArray y, NDArray? x = null, double dx = 1.0, int axis = -1)
    {
        int ax = NormalizeAxis(axis, y.Ndim);
        var yf = y.DType.IsFloat() ? y : y.AsType(DType.Float64);
        NDArray? xf = x is null ? null : x.DType.IsFloat() ? x : x.AsType(DType.Float64);
        if ((xf is null || (xf.Ndim == 1 && xf.Size == y.Shape[ax])) && KernelPath.TryTrapezoid(yf, ax, dx, xf, out var fast)) return fast;

        // the general formula, as numpy writes it
        NDArray d;
        if (xf is null) d = NDArray.WeakScalar(dx);
        else if (xf.Ndim == 1)
        {
            var shape = Enumerable.Repeat(1, y.Ndim).ToArray();
            shape[ax] = Math.Max(xf.Size - 1, 0);
            d = Reshape(Diff(xf), shape);
        }
        else d = Diff(xf, 1, ax);
        NDArray Along(Slice s) { var idx = Enumerable.Repeat<NDIndex>(Slice.All, y.Ndim).ToArray(); idx[ax] = s; return yf.Get(idx); }
        var terms = Divide(Multiply(d, Add(Along(new Slice(1, null)), Along(new Slice(null, -1)))), NDArray.WeakScalar(2.0));
        return Sum(terms, new[] { ax });
    }

    /// <summary>numpy <c>trapz</c>, the older name of <see cref="Trapezoid"/>.</summary>
    public static NDArray Trapz(NDArray y, NDArray? x = null, double dx = 1.0, int axis = -1) => Trapezoid(y, x, dx, axis);

    /// <summary>numpy <c>histogram</c> with <paramref name="bins"/> equal-width bins over <paramref name="range"/> (default: the minimum and maximum of the data).
    /// Returns the counts (int64, or float64 densities) and the <c>bins + 1</c> edges. Weights are not supported. Counting runs on the MathemaSharp kernels.</summary>
    public static (NDArray Hist, NDArray BinEdges) Histogram(NDArray a, int bins = 10, (double Min, double Max)? range = null, bool density = false)
    {
        if (bins <= 0) throw new NDValueException("`bins` must be positive, when an integer");
        double first, last;
        if (range is { } r)
        {
            (first, last) = (r.Min, r.Max);
            if (first > last) throw new NDValueException("max must be larger than min in range parameter.");
            if (!double.IsFinite(first) || !double.IsFinite(last)) throw new NDValueException($"supplied range of [{first}, {last}] is not finite");
        }
        else if (a.Size == 0) (first, last) = (0.0, 1.0);
        else
        {
            var values = (a.DType == DType.Float64 ? a : a.AsType(DType.Float64)).Copy().Data<double>();
            first = double.PositiveInfinity; last = double.NegativeInfinity;
            bool nan = false;
            foreach (var v in values) { if (double.IsNaN(v)) { nan = true; break; } if (v < first) first = v; if (v > last) last = v; }
            if (nan) first = last = double.NaN;
            if (!double.IsFinite(first) || !double.IsFinite(last)) throw new NDValueException($"autodetected range of [{first}, {last}] is not finite");
        }
        if (first == last) { first -= 0.5; last += 0.5; }
        var edges = Linspace(first, last, bins + 1).Copy().Data<double>();
        return HistogramOf(a, edges, density);
    }

    /// <summary>numpy <c>histogram</c> with explicit, non-decreasing bin edges; the last bin is closed.</summary>
    public static (NDArray Hist, NDArray BinEdges) Histogram(NDArray a, NDArray binEdges, bool density = false)
    {
        if (binEdges.Ndim != 1) throw new NDValueException("`bins` must be 1d, when an array");
        var edges = (binEdges.DType == DType.Float64 ? binEdges : binEdges.AsType(DType.Float64)).Copy().Data<double>();
        for (int i = 1; i < edges.Length; i++)
            if (edges[i - 1] > edges[i]) throw new NDValueException("`bins` must increase monotonically, when an array");
        return HistogramOf(a, edges, density);
    }

    private static (NDArray, NDArray) HistogramOf(NDArray a, double[] edges, bool density)
    {
        var counts = KernelPath.Histogram(a, edges);
        var edgeArray = NDArray.FromArray(edges, edges.Length);
        if (!density) return (NDArray.FromArray(counts, counts.Length), edgeArray);
        double total = 0;
        foreach (var c in counts) total += c;
        var dens = new double[counts.Length];
        for (int i = 0; i < dens.Length; i++) dens[i] = counts[i] / (edges[i + 1] - edges[i]) / total;   // numpy: n / db / n.sum()
        return (NDArray.FromArray(dens, dens.Length), edgeArray);
    }
}
