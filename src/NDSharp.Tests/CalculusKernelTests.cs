// Copyright (c) 2026 Marco Parenzan
// Licensed under the MIT License. See the LICENSE file in the project root.
using NDSharp;
using Xunit;

namespace NDSharp.Tests;

/// <summary>gradient and trapezoid run on the MathemaSharp kernels, histogram counts with them. For gradient and trapezoid the general formulas of the engine
/// (kernel path off) are the oracle: same dtype, shape and bits.</summary>
public class CalculusKernelTests
{
    public static IEnumerable<object[]> DtypeCases() => new[] { DType.Int32, DType.UInt8, DType.Float16, DType.Float32, DType.Float64, DType.Bool }.Select(d => new object[] { d });

    static NDArray Positions(DType dtype, int n, bool uniform)
    {
        var x = new double[n];
        double acc = 0;
        for (int i = 0; i < n; i++) { acc += uniform ? 0.5 : 0.25 + 0.1 * ((i * 7) % 5); x[i] = acc; }
        return NDArray.FromArray(x, n).AsType(dtype);
    }

    [Theory, MemberData(nameof(DtypeCases))]
    public void GradientMatchesTheEngine(DType dtype)
    {
        bool isFloat = dtype is DType.Float16 or DType.Float32 or DType.Float64;
        var f = KernelPathTests.Make(dtype, isFloat, 4, 5, 6);
        foreach (var spacing in new double[]?[] { null, [0.5], [0.1, 2.0, 0.3] })
        foreach (var axes in new int[]?[] { null, [0], [2], [1, 2], [-1] })
        {
            if (spacing is { Length: 3 } && axes is not { Length: 3 } && axes is not null) continue;
            string at = $"{dtype} spacing=[{(spacing is null ? "" : string.Join(",", spacing))}] axes=[{(axes is null ? "all" : string.Join(",", axes))}]";
            if (spacing is { Length: 3 } && axes is null) { Assert.True(RunGradient(at, () => np.Gradient(f, spacing, null)), at); continue; }
            if (spacing is { Length: 3 }) continue;
            Assert.True(RunGradient(at, () => np.Gradient(f, spacing, axes)), at);
        }
    }

    static bool RunGradient(string at, Func<NDArray[]> call)
    {
        NDArray[] expected;
        using (KernelPath.Begin(false)) expected = call();
        using var on = KernelPath.Begin(true);
        var got = call();
        Assert.Equal(expected.Length, got.Length);
        for (int i = 0; i < expected.Length; i++) KernelPathTests.SameBitsPublic(expected[i], got[i], $"gradient {at} #{i}");
        return on.Calls > 0;
    }

    [Theory, MemberData(nameof(DtypeCases))]
    public void GradientWithCoordinatesMatchesTheEngine(DType dtype)
    {
        var f = KernelPathTests.Make(dtype, false, 4, 5, 6);
        var yf = f.DType.IsFloat() ? f.DType : DType.Float64;
        foreach (var uniform in new[] { true, false })
        {
            var coords = new[] { Positions(yf, 4, uniform), Positions(yf, 5, uniform), Positions(yf, 6, uniform) };
            Assert.True(RunGradient($"{dtype} uniform={uniform}", () => np.GradientCoordinates(f, coords)), $"{dtype} uniform={uniform}");
            Assert.True(RunGradient($"{dtype} axis 1 uniform={uniform}", () => np.GradientCoordinates(f, new[] { coords[1] }, new[] { 1 })), $"{dtype} axis 1");
        }
    }

    [Fact]
    public void GradientErrorsAreUnchanged()
    {
        var one = NDArray.FromArray(new[] { 1.0 }, 1);
        Assert.Throws<NDValueException>(() => np.Gradient(one));
        var f = KernelPathTests.Make(DType.Float64, false, 4, 5);
        Assert.Throws<NDValueException>(() => np.GradientCoordinates(f, new[] { Positions(DType.Float64, 3, true), Positions(DType.Float64, 5, true) }));
    }

    [Theory, MemberData(nameof(DtypeCases))]
    public void TrapezoidMatchesTheGeneralFormula(DType dtype)
    {
        bool isFloat = dtype is DType.Float16 or DType.Float32 or DType.Float64;
        var y = KernelPathTests.Make(dtype, isFloat, 4, 5, 6);
        var yf = isFloat ? dtype : DType.Float64;
        foreach (var axis in new[] { 0, 1, 2, -1 })
        {
            int n = y.Shape[axis < 0 ? 2 : axis];
            string at = $"{dtype} axis={axis}";
            Assert.True(KernelPathTests.Compare("dx " + at, () => np.Trapezoid(y, null, 1.0, axis)), at);
            Assert.True(KernelPathTests.Compare("dx 0.37 " + at, () => np.Trapezoid(y, null, 0.37, axis)), at);
            Assert.True(KernelPathTests.Compare("x " + at, () => np.Trapezoid(y, Positions(yf, n, false), 1.0, axis)), at);
        }
        foreach (var (name, v) in new[] { ("transposed", np.Transpose(y)), ("flipped", np.Flip(y, 1)) })
            Assert.True(KernelPathTests.Compare("view " + name, () => np.Trapezoid(v, null, 0.5, 1)), name + dtype);
    }

    [Fact]
    public void TrapezoidKnownValuesAndShapes()
    {
        var y = NDArray.FromArray(new[] { 0.0, 1.0, 2.0, 3.0 }, 4);
        Assert.Equal(4.5, np.Trapezoid(y).Data<double>()[0]);
        Assert.Equal(9.0, np.Trapz(y, null, 2.0).Data<double>()[0]);
        Assert.Equal(6.0, np.Trapezoid(NDArray.FromArray(new[] { 1.0, 1.0, 1.0, 1.0 }, 4), NDArray.FromArray(new[] { 0.0, 1.0, 3.0, 6.0 }, 4)).Data<double>()[0]);
        Assert.Equal(0.0, np.Trapezoid(NDArray.FromArray(new[] { 5.0 }, 1)).Data<double>()[0]);
        var m = NDArray.FromArray(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }, 2, 3);
        Assert.Equal(new[] { 2 }, np.Trapezoid(m, null, 1.0, 1).Shape);
        Assert.Equal(new[] { 3 }, np.Trapezoid(m, null, 1.0, 0).Shape);
        Assert.Equal(new[] { 4.0, 10.0 }, np.Trapezoid(m, null, 1.0, 1).Copy().Data<double>());
    }

    static long[] Reference(double[] data, double[] edges)
    {
        var c = new long[edges.Length - 1];
        foreach (var v in data)
        {
            if (double.IsNaN(v) || v < edges[0] || v > edges[^1]) continue;
            int b = c.Length - 1;
            for (int i = 0; i < c.Length; i++) if (v < edges[i + 1]) { b = i; break; }
            c[b]++;
        }
        return c;
    }

    [Fact]
    public void HistogramCountsLikeAReferenceLoop()
    {
        var data = Enumerable.Range(0, 500).Select(i => Math.Round(Math.Sin(i * 0.13) * 40 + (i % 7), 2)).ToArray();
        data[10] = double.NaN;
        var a = NDArray.FromArray(data, 25, 20);
        var (hist, edges) = np.Histogram(a, new NDArray[] { }.Length + 7, (-50, 50));
        Assert.Equal(DType.Int64, hist.DType);
        Assert.Equal(Reference(data, edges.Copy().Data<double>()), hist.Copy().Data<long>());
        Assert.Equal(8, edges.Size);
        var (h2, e2) = np.Histogram(a, NDArray.FromArray(new[] { -100.0, -10.0, 0.0, 0.0, 5.0, 100.0 }, 6));
        Assert.Equal(Reference(data, e2.Copy().Data<double>()), h2.Copy().Data<long>());
    }

    [Fact]
    public void HistogramNumpyRules()
    {
        // numpy: np.histogram([1, 2, 1], bins=[0, 1, 2, 3]) -> [0, 2, 1]; the last bin is closed
        var (h, _) = np.Histogram(NDArray.FromArray(new[] { 1.0, 2.0, 1.0 }, 3), NDArray.FromArray(new[] { 0.0, 1.0, 2.0, 3.0 }, 4));
        Assert.Equal(new long[] { 0, 2, 1 }, h.Copy().Data<long>());
        var (h3, e3) = np.Histogram(NDArray.FromArray(new[] { 3.0, 3.0, 3.0 }, 3), 2);   // a constant: the range widens by 0.5 each side
        Assert.Equal(new[] { 2.5, 3.0, 3.5 }, e3.Copy().Data<double>());
        Assert.Equal(new long[] { 0, 3 }, h3.Copy().Data<long>());
        var (hi, ei) = np.Histogram(NDArray.FromArray(new[] { 1, 2, 3, 4, 5 }, 5), 2);   // integers: range 1..5
        Assert.Equal(new[] { 1.0, 3.0, 5.0 }, ei.Copy().Data<double>());
        Assert.Equal(new long[] { 2, 3 }, hi.Copy().Data<long>());
        var (empty, ee) = np.Histogram(NDArray.FromArray(Array.Empty<double>(), 0), 4);
        Assert.Equal(new long[] { 0, 0, 0, 0 }, empty.Copy().Data<long>());
        Assert.Equal(new[] { 0.0, 0.25, 0.5, 0.75, 1.0 }, ee.Copy().Data<double>());
    }

    [Fact]
    public void HistogramDensityAndErrors()
    {
        var a = NDArray.FromArray(new[] { 0.5, 1.5, 1.5, 2.5 }, 4);
        var (d, _) = np.Histogram(a, NDArray.FromArray(new[] { 0.0, 1.0, 2.0, 4.0 }, 4), density: true);
        Assert.Equal(new[] { 0.25, 0.5, 0.125 }, d.Copy().Data<double>());   // counts 1, 2, 1 over widths 1, 1, 2 and a total of 4
        Assert.Throws<NDValueException>(() => np.Histogram(a, 0));
        Assert.Throws<NDValueException>(() => np.Histogram(a, 3, (5.0, 1.0)));
        Assert.Throws<NDValueException>(() => np.Histogram(NDArray.FromArray(new[] { 1.0, double.NaN }, 2), 3));
        Assert.Throws<NDValueException>(() => np.Histogram(a, NDArray.FromArray(new[] { 2.0, 1.0 }, 2)));
        Assert.Throws<NDValueException>(() => np.Histogram(a, 3, (0.0, double.PositiveInfinity)));
    }
}
