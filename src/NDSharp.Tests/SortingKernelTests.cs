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
            Assert.True(KernelPathTests.Compare($"sort {name} {dtype}", () => np.Sort(v, 1)), name);
            Assert.True(KernelPathTests.Compare($"argsort {name} {dtype}", () => np.ArgSort(v, 0)), name);
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
        // complex numbers sort too, in numpy's lexicographic order
        using var scope = KernelPath.Begin();
        _ = np.Sort(np.Zeros(new[] { 4 }, DType.Complex128));
        Assert.Equal(1, scope.Calls);
    }

    [Fact]
    public void SortKnownValues()
    {
        using var _ = KernelPath.Begin();
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
            Assert.True(KernelPathTests.CompareMany($"unique {dtype} index={ri} inverse={rv} counts={rc}", () => { var u = np.Unique(a, ri, rv, rc); return new[] { u.Values, u.Index!, u.Inverse!, u.Counts! }.Where(x => x is not null).ToArray(); }));
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
        var days = NDArray.FromArray(new[] { 1.0 }, 1).AsType(DType.DateTime64D);
        Assert.Throws<NDNotSupportedException>(() => np.SearchSorted(days, days));   // dates are not searched here
    }

    // ------------------------------------------------------------------ complex, partition, lexsort

    static NDArray ComplexData(DType dtype, params int[] shape)
    {
        int n = 1; foreach (var x in shape) n *= x;
        var d = new System.Numerics.Complex[n];
        for (int i = 0; i < n; i++) d[i] = new System.Numerics.Complex(((i * 7) % 5) - 2, ((i * 3) % 4) - 1);   // many repeated values
        if (n > 20) { d[2] = new(double.NaN, 1); d[5] = new(1, double.NaN); d[9] = new(double.NaN, double.NaN); d[11] = new(double.NaN, -3); d[14] = new(2, double.NaN); }
        return NDArray.FromArray(d, shape).AsType(dtype);
    }

    [Theory]
    [InlineData(DType.Complex128)]
    [InlineData(DType.Complex64)]
    public void ComplexSortArgSortAndUniqueMatchTheEngine(DType dtype)
    {
        var a = ComplexData(dtype, 4, 5, 6);
        foreach (int? axis in new int?[] { null, 0, 1, 2, -1 })
        {
            Assert.True(KernelPathTests.Compare($"sort {dtype} axis={axis}", () => np.Sort(a, axis)));
            Assert.True(KernelPathTests.Compare($"argsort {dtype} axis={axis}", () => np.ArgSort(a, axis)));
        }
        Assert.True(KernelPathTests.Compare($"sort transposed {dtype}", () => np.Sort(np.Transpose(a), 1)));
        Assert.True(KernelPathTests.CompareMany($"unique {dtype}", () => { var u = np.Unique(a, true, true, true); return new[] { u.Values, u.Index!, u.Inverse!, u.Counts! }; }));
    }

    [Fact]
    public void ComplexSearchSorted()
    {
        var sorted = NDArray.FromArray(new[] { new System.Numerics.Complex(0, 0), new(0, 5), new(0, 5), new(1, 1), new(1, double.NaN), new(double.NaN, 0) }, 6);
        var values = NDArray.FromArray(new[] { new System.Numerics.Complex(0, 5), new(0, 6), new(2, 0), new(1, double.NaN), new(double.NaN, 1) }, 5);
        Assert.Equal(new long[] { 1, 3, 4, 4, 6 }, np.SearchSorted(sorted, values).Copy().Data<long>());
        Assert.Equal(new long[] { 3, 3, 4, 5, 6 }, np.SearchSorted(sorted, values, "right").Copy().Data<long>());
    }

    [Fact]
    public void PartitionKeepsTheKthElementAndTheOrderAroundIt()
    {
        var rnd = new System.Random(11);
        var data = Enumerable.Range(0, 3 * 40).Select(_ => (double)rnd.Next(-20, 20)).ToArray();
        data[5] = double.NaN; data[70] = double.NaN;
        var a = NDArray.FromArray(data, 3, 40);
        foreach (var kth in new[] { new[] { 0 }, new[] { 7 }, new[] { -1 }, new[] { 3, 20, 39 }, new[] { -40 } })
        {
            using var on = KernelPath.Begin();
            var p = np.Partition(a, kth, 1);
            var sorted = np.Sort(a, 1).Copy().Data<double>();
            var got = p.Copy().Data<double>();
            for (int row = 0; row < 3; row++)
                foreach (var k in kth)
                {
                    int pos = (k + 40) % 40;
                    var line = got.AsSpan(row * 40, 40);
                    double pivot = sorted[row * 40 + pos];
                    Assert.True(line[pos].Equals(pivot));
                    for (int i = 0; i < 40; i++)
                    {
                        if (i < pos) Assert.True(double.IsNaN(pivot) || (!double.IsNaN(line[i]) && line[i] <= pivot));    // nothing larger before it (NaN is the largest)
                        if (i > pos) Assert.True(double.IsNaN(line[i]) || (!double.IsNaN(pivot) && line[i] >= pivot));    // nothing smaller after it
                    }
                }
        }
        Assert.Throws<NDValueException>(() => np.Partition(a, 40, 1));
        Assert.Throws<NDValueException>(() => np.Partition(a, -41, 1));
        Assert.Throws<NDValueException>(() => np.ArgPartition(a, new[] { 0, 99 }, 1));
    }

    [Fact]
    public void ArgPartitionShapesAndFlattened()
    {
        var a = NDArray.FromArray(new[] { 5.0, 1.0, 4.0, 2.0, 3.0, 9.0 }, 3, 2);
        Assert.Equal(new[] { 3, 2 }, np.ArgPartition(a, 1, 0).Shape);
        Assert.Equal(new[] { 6 }, np.ArgPartition(a, 2, null).Shape);
        Assert.Equal(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 9.0 }, np.Partition(a, 2, null).Copy().Data<double>());
        Assert.Equal(new long[] { 0, 1, 2 }, np.ArgPartition(NDArray.FromArray(new[] { 1, 2, 3 }, 3), 1).Copy().Data<long>());
    }

    static long[] LexReference(double[][] keys, int n)
    {
        // the last key is primary: stable sort by each key from the first to the last
        IEnumerable<int> ordered = Enumerable.Range(0, n);
        foreach (var key in keys) ordered = ordered.OrderBy(i => key[i]);
        return ordered.Select(i => (long)i).ToArray();
    }

    [Fact]
    public void LexSortOneDimensionalKeys()
    {
        var rnd = new System.Random(5);
        double[] a = Enumerable.Range(0, 60).Select(_ => (double)rnd.Next(0, 4)).ToArray();
        double[] b = Enumerable.Range(0, 60).Select(_ => (double)rnd.Next(0, 3)).ToArray();
        double[] c = Enumerable.Range(0, 60).Select(_ => (double)rnd.Next(0, 5)).ToArray();
        using var on = KernelPath.Begin();
        var keys = new List<NDArray> { NDArray.FromArray(a, 60), NDArray.FromArray(b, 60), NDArray.FromArray(c, 60) };
        Assert.Equal(LexReference(new[] { a, b, c }, 60), np.LexSort(keys).Copy().Data<long>());
        Assert.Equal(LexReference(new[] { a, b, c }, 60), np.LexSort(NDArray.FromArray(a.Concat(b).Concat(c).ToArray(), 3, 60)).Copy().Data<long>());
        // sorted by the second key (the primary), ties by the first: keys (1,5,1,4,3,4,4) and (9,4,0,4,0,2,1)
        var ind = np.LexSort(new List<NDArray> { NDArray.FromArray(new[] { 1.0, 5.0, 1.0, 4.0, 3.0, 4.0, 4.0 }, 7), NDArray.FromArray(new[] { 9.0, 4.0, 0.0, 4.0, 0.0, 2.0, 1.0 }, 7) });
        Assert.Equal(new long[] { 2, 4, 6, 5, 3, 1, 0 }, ind.Copy().Data<long>());
    }

    [Fact]
    public void LexSortAlongAnAxisAndErrors()
    {
        var k1 = NDArray.FromArray(new[] { 2.0, 1.0, 1.0, 0.0, 5.0, 5.0 }, 2, 3);
        var k2 = NDArray.FromArray(new[] { 0.0, 0.0, 0.0, 9.0, 8.0, 7.0 }, 2, 3);
        var r = np.LexSort(new List<NDArray> { k1, k2 }, 1);           // each row: primary k2, then k1
        Assert.Equal(new[] { 2, 3 }, r.Shape);
        Assert.Equal(new long[] { 1, 2, 0, 2, 1, 0 }, r.Copy().Data<long>());
        Assert.Equal(new long[] { 1, 0 }, np.LexSort(new List<NDArray> { NDArray.FromArray(new[] { 2.0, 1.0 }, 2) }).Copy().Data<long>());
        Assert.Throws<NDValueException>(() => np.LexSort(new List<NDArray>()));
        Assert.Throws<NDValueException>(() => np.LexSort(new List<NDArray> { k1, NDArray.FromArray(new[] { 1.0, 2.0 }, 2) }));
        Assert.Empty(np.LexSort(new List<NDArray> { NDArray.FromArray(Array.Empty<double>(), 0) }).Copy().Data<long>());
    }
}
