// Copyright (c) 2026 Marco Parenzan
// Licensed under the MIT License. See the LICENSE file in the project root.
using NDSharp;
using Xunit;

namespace NDSharp.Tests;

/// <summary>sort, argsort and unique run on the MathemaSharp kernels (the engine is the oracle: same dtype, shape and bits); searchsorted is new and is checked against a brute-force count.</summary>
public class SortingKernelTests
{
    public static IEnumerable<object[]> DtypeCases() => KernelPathTests.Dtypes.Select(d => new object[] { d });

    /// <summary>Data with many repeated values (so that argsort stability matters) and, for floats, NaN, infinities and both zeros.</summary>
    static NDArray Repeated(DType dtype, params int[] shape)
    {
        int n = 1; foreach (var s in shape) n *= s;
        var d = new double[n];
        for (int i = 0; i < n; i++) d[i] = ((i * 7919) % 11) - 5;
        if (n > 30 && dtype is DType.Float16 or DType.Float32 or DType.Float64)
        {
            d[3] = double.NaN; d[9] = double.NaN; d[14] = double.PositiveInfinity; d[15] = double.NegativeInfinity; d[20] = -0.0; d[21] = 0.0; d[26] = -0.0;
        }
        return NDArray.FromArray(d, shape).AsType(dtype);
    }

    [Theory, MemberData(nameof(DtypeCases))]
    public void SortAndArgSortMatchTheEngine(DType dtype)
    {
        var a = Repeated(dtype, 4, 5, 6);
        foreach (int? axis in new int?[] { null, 0, 1, 2, -1 })
        {
            Assert.True(KernelPathTests.Compare($"sort {dtype} axis={axis}", () => np.Sort(a, axis)));
            Assert.True(KernelPathTests.Compare($"argsort {dtype} axis={axis}", () => np.ArgSort(a, axis)));
        }
        foreach (var (name, v) in new[] { ("transposed", np.Transpose(a)), ("flipped", np.Flip(a, 1)), ("moved", np.MoveAxis(a, 0, 2)) })
        {
            Assert.True(KernelPathTests.Compare($"sort {name}", () => np.Sort(v, 1)), name);
            Assert.True(KernelPathTests.Compare($"argsort {name}", () => np.ArgSort(v, 0)), name);
        }
    }

    [Fact]
    public void SortEdgeCases()
    {
        Assert.True(KernelPathTests.Compare("one element", () => np.Sort(NDArray.FromArray(new[] { 3.0 }, 1))));
        Assert.True(KernelPathTests.Compare("empty", () => np.Sort(NDArray.FromArray(Array.Empty<double>(), 0))));
        Assert.True(KernelPathTests.Compare("empty rows", () => np.Sort(np.Zeros(new[] { 0, 3 }, DType.Float64), 0)));
        Assert.True(KernelPathTests.Compare("argsort empty", () => np.ArgSort(NDArray.FromArray(Array.Empty<double>(), 0))));
        KernelPathTests.Compare("0-d", () => np.Sort(NDArray.Scalar(1.0)));
        KernelPathTests.Compare("bad axis", () => np.Sort(Repeated(DType.Int32, 3, 3), 2));
        // complex stays on the engine (numpy's lexicographic order is not a real-number order)
        using var scope = KernelPath.Begin(true);
        _ = np.Sort(np.Zeros(new[] { 4 }, DType.Complex128));
        Assert.Equal(0, scope.Calls);
    }

    [Fact]
    public void SortKnownValues()
    {
        using var _ = KernelPath.Begin(true);
        Assert.Equal(new[] { 1.0, 2.0, 3.0, double.NaN }, np.Sort(NDArray.FromArray(new[] { 3.0, double.NaN, 1.0, 2.0 }, 4)).Copy().Data<double>());
        Assert.Equal(new long[] { 2, 3, 0, 1 }, np.ArgSort(NDArray.FromArray(new[] { 3.0, double.NaN, 1.0, 2.0 }, 4)).Copy().Data<long>());
        Assert.Equal(new long[] { 1, 3, 0, 2 }, np.ArgSort(NDArray.FromArray(new[] { 2, 1, 2, 1 }, 4)).Copy().Data<long>());   // stable
    }

    [Theory]
    [InlineData(DType.Int32)]
    [InlineData(DType.Float64)]
    [InlineData(DType.Float32)]
    [InlineData(DType.Bool)]
    [InlineData(DType.UInt8)]
    public void UniqueMatchesTheEngine(DType dtype)
    {
        var a = Repeated(dtype, 5, 8);
        foreach (var (ri, rv, rc) in new[] { (false, false, false), (true, true, true), (true, false, false), (false, true, false) })
        {
            NDArray[] Parts() { var u = np.Unique(a, ri, rv, rc); return new[] { u.Values, u.Index!, u.Inverse!, u.Counts! }.Where(x => x is not null).ToArray(); }
            NDArray[] expected;
            using (KernelPath.Begin(false)) expected = Parts();
            using var on = KernelPath.Begin(true);
            var got = Parts();
            Assert.Equal(expected.Length, got.Length);
            for (int i = 0; i < expected.Length; i++) KernelPathTests.SameBitsPublic(expected[i], got[i], $"unique {dtype} #{i}");
            Assert.True(on.Calls > 0);
        }
    }

    // ------------------------------------------------------------------ searchsorted

    static bool Less(double x, double v) => !double.IsNaN(x) && (double.IsNaN(v) || x < v);        // x before v in the order with NaN last
    static bool LessOrEqual(double x, double v) => double.IsNaN(v) || (!double.IsNaN(x) && x <= v);

    [Fact]
    public void SearchSortedAgreesWithACount()
    {
        var sorted = new[] { -2.0, -2.0, 0.0, 1.5, 1.5, 1.5, 4.0, double.PositiveInfinity, double.NaN, double.NaN };
        var values = new[] { double.NegativeInfinity, -3.0, -2.0, -1.0, 0.0, 1.5, 3.0, 4.0, 5.0, double.PositiveInfinity, double.NaN };
        foreach (var side in new[] { "left", "right" })
        {
            var got = np.SearchSorted(NDArray.FromArray(sorted, sorted.Length), NDArray.FromArray(values, values.Length), side).Copy().Data<long>();
            for (int i = 0; i < values.Length; i++)
            {
                long expected = side == "left" ? sorted.Count(x => Less(x, values[i])) : sorted.Count(x => LessOrEqual(x, values[i]));
                Assert.Equal(expected, got[i]);
            }
        }
    }

    [Fact]
    public void SearchSortedShapesDtypesAndSorter()
    {
        var a = NDArray.FromArray(new[] { 1, 3, 5, 7, 9 }, 5);
        var v2d = NDArray.FromArray(new[] { 0, 3, 4, 10 }, 2, 2);
        var r = np.SearchSorted(a, v2d);
        Assert.Equal(new[] { 2, 2 }, r.Shape);
        Assert.Equal(new long[] { 0, 1, 2, 5 }, r.Copy().Data<long>());
        Assert.Equal(new long[] { 0, 2, 2, 5 }, np.SearchSorted(a, v2d, "right").Copy().Data<long>());
        var scalar = np.SearchSorted(a, NDArray.Scalar(2.5));          // mixed int and float: searched as float64; a scalar value gives a scalar index
        Assert.Empty(scalar.Shape);
        Assert.Equal(1L, scalar.Copy().Data<long>()[0]);
        var unsorted = NDArray.FromArray(new[] { 9.0, 1.0, 7.0, 3.0, 5.0 }, 5);
        var sorter = np.ArgSort(unsorted);
        Assert.Equal(new long[] { 0, 2, 5 }, np.SearchSorted(unsorted, NDArray.FromArray(new[] { 0.5, 3.5, 11.0 }, 3), "left", sorter).Copy().Data<long>());
        var flags = NDArray.FromArray(new[] { false, true, true }, 3);                  // bool is searched as uint8
        Assert.Equal(1L, np.SearchSorted(flags, NDArray.FromArray(new[] { true }, 1), "left").Copy().Data<long>()[0]);
        Assert.Equal(3L, np.SearchSorted(flags, NDArray.FromArray(new[] { true }, 1), "right").Copy().Data<long>()[0]);
        Assert.Equal(new long[] { 0, 0 }, np.SearchSorted(NDArray.FromArray(Array.Empty<double>(), 0), NDArray.FromArray(new[] { 1.0, double.NaN }, 2)).Copy().Data<long>());
    }

    [Fact]
    public void SearchSortedErrors()
    {
        var a = NDArray.FromArray(new[] { 1.0, 2.0 }, 2);
        Assert.Throws<NDValueException>(() => np.SearchSorted(a, a, "middle"));
        Assert.Throws<NDValueException>(() => np.SearchSorted(np.Zeros(new[] { 2, 2 }, DType.Float64), a));
        Assert.Throws<NDValueException>(() => np.SearchSorted(a, a, "left", NDArray.FromArray(new long[] { 0 }, 1)));
        Assert.Throws<NDNotSupportedException>(() => np.SearchSorted(np.Zeros(new[] { 2 }, DType.Complex128), a));
    }
}
