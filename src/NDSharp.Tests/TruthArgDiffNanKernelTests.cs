// Copyright (c) 2026 Marco Parenzan
// Licensed under the MIT License. See the LICENSE file in the project root.
using NDSharp;
using Xunit;

namespace NDSharp.Tests;

/// <summary>any, all, count_nonzero, argmax, argmin and diff run on the MathemaSharp kernels (the general engine is the oracle: same dtype, shape and bits); the nan* functions
/// are checked against numpy's own formulas written with elementary operations.</summary>
public class TruthArgDiffNanKernelTests
{
    public static IEnumerable<object[]> DtypeCases() => KernelPathTests.Dtypes.Select(d => new object[] { d });

    /// <summary>Data with zeros (and, for floats, NaN, infinity and negative zero) so that the truth functions have something to decide.</summary>
    static NDArray Sparse(DType dtype, params int[] shape)
    {
        int n = 1; foreach (var s in shape) n *= s;
        var d = new double[n];
        for (int i = 0; i < n; i++) d[i] = (i % 5 == 0 || i % 7 == 3) ? 0 : Math.Round(Math.Sin(i * 0.9) * 20, 2);
        if (n > 40) { d[2] = d[3] = d[4] = 0; d[10] = double.NaN; d[17] = double.PositiveInfinity; d[23] = -0.0; }
        // one lane that is all zero along the last axis, one that is all non-zero
        for (int i = 6; i < 12 && i < n; i++) d[i] = i == 10 ? double.NaN : 0;
        return NDArray.FromArray(d, shape).AsType(dtype);
    }

    [Theory, MemberData(nameof(DtypeCases))]
    public void AnyAllAndCountMatchTheEngine(DType dtype)
    {
        var a = Sparse(dtype, 4, 5, 6);
        foreach (var axes in KernelPathTests.AxisSets)
        foreach (var keep in new[] { false, true })
        {
            string at = $"{dtype} axes=[{(axes is null ? "all" : string.Join(",", axes))}] keep={keep}";
            Assert.True(KernelPathTests.Compare("any " + at, () => np.Any(a, axes, keep)), at);
            Assert.True(KernelPathTests.Compare("all " + at, () => np.All(a, axes, keep)), at);
            Assert.True(KernelPathTests.Compare("count_nonzero " + at, () => np.CountNonzero(a, axes, keep)), at);
        }
        foreach (var (name, v) in new[] { ("transposed", np.Transpose(a)), ("flipped", np.Flip(a, 1)) })
        {
            Assert.True(KernelPathTests.Compare("any " + name, () => np.Any(v, [1])), name);
            Assert.True(KernelPathTests.Compare("count_nonzero " + name, () => np.CountNonzero(v, [0, 2])), name);
        }
    }

    [Theory, MemberData(nameof(DtypeCases))]
    public void ArgMaxAndArgMinMatchTheEngine(DType dtype)
    {
        bool isFloat = dtype is DType.Float16 or DType.Float32 or DType.Float64;
        foreach (var specials in isFloat ? new[] { false, true } : [false])
        {
            var a = KernelPathTests.Make(dtype, specials, 4, 5, 6);
            foreach (int? axis in new int?[] { null, 0, 1, 2, -1 })
            foreach (var keep in new[] { false, true })
            {
                string at = $"{dtype} specials={specials} axis={axis} keep={keep}";
                Assert.True(KernelPathTests.Compare("argmax " + at, () => np.ArgMax(a, axis, keep)), at);
                Assert.True(KernelPathTests.Compare("argmin " + at, () => np.ArgMin(a, axis, keep)), at);
            }
            Assert.True(KernelPathTests.Compare("argmax transposed", () => np.ArgMax(np.Transpose(a), 1)));
            Assert.True(KernelPathTests.Compare("argmin flipped", () => np.ArgMin(np.Flip(a, 2), null)));
        }
    }

    [Fact]
    public void ArgExtremeTiesNanAndEmpty()
    {
        var tie = NDArray.FromArray(new[] { 3.0, 9.0, 9.0, 1.0, 1.0 }, 5);
        using (KernelPath.Begin(true))
        {
            Assert.Equal(1L, np.ArgMax(tie).Data<long>()[0]);   // the first of equal elements
            Assert.Equal(3L, np.ArgMin(tie).Data<long>()[0]);
            Assert.Equal(1L, np.ArgMax(NDArray.FromArray(new[] { 1.0, double.NaN, 5.0 }, 3)).Data<long>()[0]);   // the first NaN wins
        }
        KernelPathTests.Compare("argmax empty", () => np.ArgMax(NDArray.FromArray(Array.Empty<double>(), 0)));
        KernelPathTests.Compare("argmin empty axis", () => np.ArgMin(NDArray.FromArray(Array.Empty<double>(), 0, 3), 0));
        KernelPathTests.Compare("argmax bad axis", () => np.ArgMax(tie, 2));
    }

    [Theory, MemberData(nameof(DtypeCases))]
    public void DiffMatchesTheEngine(DType dtype)
    {
        var a = KernelPathTests.Make(dtype, dtype is DType.Float16 or DType.Float32 or DType.Float64, 4, 5, 6);
        bool kernel = dtype != DType.Bool;   // bool differences are comparisons, left to the engine
        foreach (var n in new[] { 1, 2, 3, 4, 5, 9 })
        foreach (var axis in new[] { 0, 1, 2, -1 })
        {
            string at = $"{dtype} n={n} axis={axis}";
            bool used = KernelPathTests.Compare("diff " + at, () => np.Diff(a, n, axis));
            Assert.Equal(kernel, used);
        }
        if (kernel) Assert.True(KernelPathTests.Compare("diff transposed", () => np.Diff(np.Transpose(a), 2, 1)));
    }

    [Fact]
    public void DiffKnownValuesAndShapes()
    {
        using var _ = KernelPath.Begin(true);
        Assert.Equal(new[] { 3.0, 5.0, 7.0 }, np.Diff(NDArray.FromArray(new[] { 1.0, 4.0, 9.0, 16.0 }, 4)).Copy().Data<double>());
        Assert.Equal(new[] { 2.0, 2.0 }, np.Diff(NDArray.FromArray(new[] { 1.0, 4.0, 9.0, 16.0 }, 4), 2).Copy().Data<double>());
        Assert.Equal(new byte[] { 255, 1 }, np.Diff(NDArray.FromArray(new byte[] { 5, 4, 5 }, 3)).Copy().Data<byte>());   // unsigned wrap, as numpy
        Assert.Equal(new[] { 2, 0 }, np.Diff(NDArray.FromArray(new[] { 1, 2, 3, 4, 5, 6, 7, 8 }, 2, 4), 4, 1).Shape);   // n as long as the axis: nothing is left
        Assert.Equal(0, np.Diff(NDArray.FromArray(new[] { 1.0, 2.0 }, 2), 5).Size);
    }

    // ------------------------------------------------------------------ nan*

    /// <summary>A 4 x 5 x 6 array with a NaN every third element and all-NaN lanes along each axis.</summary>
    static NDArray ManyNans(DType dtype, bool allNanLanes)
    {
        var d = new double[120];
        for (int i = 0; i < 120; i++) d[i] = (i % 3 == 1) ? double.NaN : Math.Round(Math.Sin((i + 1) * 0.41) * 30 + (i % 7), 3);
        if (!allNanLanes) EnsureEveryLaneHasAValue(d);
        if (allNanLanes)
        {
            for (int i = 0; i < 6; i++) d[i] = double.NaN;               // along axis 2
            for (int i = 0; i < 120; i += 30) d[i] = double.NaN;         // along axis 0
            for (int b = 0; b < 5; b++) d[1 + b * 6] = double.NaN;       // along axis 1
        }
        return NDArray.FromArray(d, 4, 5, 6).AsType(dtype);
    }

    /// <summary>A NaN every third element makes whole lanes NaN along axes 0 and 1; give each lane (shape 4 x 5 x 6) at least one value.</summary>
    static void EnsureEveryLaneHasAValue(double[] d)
    {
        for (int c = 0; c < 6; c++)
        {
            for (int b = 0; b < 5; b++)   // lanes along axis 0
                if (Enumerable.Range(0, 4).All(a => double.IsNaN(d[a * 30 + b * 6 + c]))) d[b * 6 + c] = 1.5 + b + c;
            for (int a = 0; a < 4; a++)   // lanes along axis 1
                if (Enumerable.Range(0, 5).All(b => double.IsNaN(d[a * 30 + b * 6 + c]))) d[a * 30 + c] = 2.5 + a + c;
        }
    }

    static NDArray IsNanOf(NDArray a) => np.IsNan(a);

    static NDArray Replace(NDArray a, double with) => np.Where(IsNanOf(a), NDArray.WeakScalar(with), a);

    static readonly DType[] FloatTypes = [DType.Float16, DType.Float32, DType.Float64];
    public static IEnumerable<object[]> FloatCases() => FloatTypes.Select(d => new object[] { d });

    static void Same(NDArray expected, NDArray actual, string what) => KernelPathTests.SameBitsPublic(expected, actual, what);

    [Theory, MemberData(nameof(FloatCases))]
    public void NanSumProdMaxMinMatchNumpysFormulas(DType dtype)
    {
        foreach (var allNan in new[] { false, true })
        {
            var a = ManyNans(dtype, allNan);
            foreach (var axes in KernelPathTests.AxisSets)
            foreach (var keep in new[] { false, true })
            {
                string at = $"{dtype} allNan={allNan} axes=[{(axes is null ? "all" : string.Join(",", axes))}] keep={keep}";
                NDArray sumExpected, prodExpected, maxExpected, minExpected;
                using (KernelPath.Begin(false))
                {
                    sumExpected = np.Sum(Replace(a, 0), axes, keep);
                    prodExpected = np.Prod(Replace(a, 1), axes, keep);
                    var anyValid = np.Any(np.LogicalNot(IsNanOf(a)), axes, keep);
                    maxExpected = np.Where(anyValid, np.Max(Replace(a, double.NegativeInfinity), axes, keep), NDArray.WeakScalar(double.NaN));
                    minExpected = np.Where(anyValid, np.Min(Replace(a, double.PositiveInfinity), axes, keep), NDArray.WeakScalar(double.NaN));
                }
                using (KernelPath.Begin(false))   // the nan* functions ignore the switch: they always run on the kernels
                {
                    Same(sumExpected, np.NanSum(a, axes, keep), "nansum " + at);
                    Same(prodExpected, np.NanProd(a, axes, keep), "nanprod " + at);
                    Same(maxExpected, np.NanMax(a, axes, keep), "nanmax " + at);
                    Same(minExpected, np.NanMin(a, axes, keep), "nanmin " + at);
                }
            }
        }
    }

    [Theory, MemberData(nameof(FloatCases))]
    public void NanMeanVarStdMatchNumpysFormulas(DType dtype)
    {
        foreach (var allNan in new[] { false, true })
        {
            var a = ManyNans(dtype, allNan);
            foreach (var axes in KernelPathTests.AxisSets)
            {
                foreach (var ddof in new[] { 0, 1 })
                {
                    string at = $"{dtype} allNan={allNan} axes=[{(axes is null ? "all" : string.Join(",", axes))}] ddof={ddof}";
                    NDArray meanExpected, varExpected;
                    using (KernelPath.Begin(false))
                    {
                        var arr = Replace(a, 0);
                        var cnt = np.Sum(np.LogicalNot(IsNanOf(a)), axes, false);                  // int64
                        meanExpected = np.Divide(np.Sum(arr, axes, false), cnt).AsType(dtype);       // numpy: tot / cnt, cast back to the array's dtype
                        var cntK = np.Sum(np.LogicalNot(IsNanOf(a)), axes, true);
                        var avg = np.Divide(np.Sum(arr, axes, true), cntK).AsType(dtype);
                        var dev = np.Where(IsNanOf(a), NDArray.WeakScalar(0.0), np.Subtract(arr, avg));
                        var dof = np.Subtract(cnt, NDArray.WeakScalar((long)ddof));
                        var bad = np.LessEqual(dof, NDArray.WeakScalar(0L));
                        var dofF = np.Where(bad, NDArray.WeakScalar(double.NaN), dof.AsType(DType.Float64));
                        varExpected = np.Divide(np.Sum(np.Multiply(dev, dev), axes, false), dofF).AsType(dtype);
                    }
                    Same(meanExpected, np.NanMean(a, axes, false), "nanmean " + at);
                    Same(varExpected, np.NanVar(a, axes, false, ddof), "nanvar " + at);
                    using (KernelPath.Begin(false)) Same(np.Sqrt(varExpected), np.NanStd(a, axes, false, ddof), "nanstd " + at);
                }
            }
        }
    }

    /// <summary>The lanes of <paramref name="a"/> along <paramref name="axis"/> as plain double arrays, in C order of the other axes.</summary>
    static List<double[]> Lanes(NDArray a, int axis)
    {
        var data = np.MoveAxis(a, axis, a.Ndim - 1).Copy().AsType(DType.Float64).Copy().Data<double>();
        int n = a.Shape[axis];
        return Enumerable.Range(0, data.Length / n).Select(i => data.AsSpan(i * n, n).ToArray()).ToList();
    }

    static NDArray WithoutNan(double[] lane, DType dtype)
    {
        var kept = lane.Where(v => !double.IsNaN(v)).ToArray();
        return NDArray.FromArray(kept, kept.Length).AsType(dtype);
    }

    [Theory, MemberData(nameof(FloatCases))]
    public void NanMedianPercentileAndArgsMatchPerLaneReferences(DType dtype)
    {
        if (dtype == DType.Float16) return;   // the per-lane reference below uses the engine's median, which float16 shares through the kernels: covered by the float32/float64 cases
        var a = ManyNans(dtype, false);
        foreach (var axis in new[] { 0, 1, 2 })
        {
            var lanes = Lanes(a, axis);
            using (KernelPath.Begin(false))
            {
                var nm = np.NanMedian(a, [axis]);
                var np30 = np.NanPercentile(a, NDArray.Scalar(30.0), [axis]);
                var narg = np.NanArgMax(a, axis);
                var nmFlat = nm.Copy().AsType(DType.Float64).Copy().Data<double>();
                var np30Flat = np30.Copy().AsType(DType.Float64).Copy().Data<double>();
                var nargFlat = narg.Copy().Data<long>();
                for (int i = 0; i < lanes.Count; i++)
                {
                    var clean = WithoutNan(lanes[i], dtype);
                    Assert.Equal(Convert.ToDouble(np.Median(clean).GetAt(0)), nmFlat[i]);
                    Assert.Equal(Convert.ToDouble(np.Percentile(clean, NDArray.Scalar(30.0)).GetAt(0)), np30Flat[i]);
                    var valid = Enumerable.Range(0, lanes[i].Length).Where(k => !double.IsNaN(lanes[i][k])).ToArray();
                    Assert.Equal(valid[(int)np.ArgMax(clean).Data<long>()[0]], (int)nargFlat[i]);   // the index in the full lane
                }
                Assert.Equal(a.Shape.Where((_, i) => i != axis).ToArray(), nm.Shape);
            }
        }
    }

    [Fact]
    public void NanArgExtremeFailsOnAnAllNanSlice()
    {
        var a = ManyNans(DType.Float64, true);
        Assert.Throws<NDValueException>(() => np.NanArgMax(a, 2));
        Assert.Throws<NDValueException>(() => np.NanArgMin(a, 0));
        var fine = ManyNans(DType.Float64, false);
        Assert.Equal(new[] { 4, 6 }, np.NanArgMax(fine, 1).Shape);
    }

    [Theory, MemberData(nameof(FloatCases))]
    public void NanCumulativeMatchesNumpysFormula(DType dtype)
    {
        var a = ManyNans(dtype, true);
        foreach (int? axis in new int?[] { null, 0, 1, 2, -1 })
        {
            NDArray sumExpected, prodExpected;
            using (KernelPath.Begin(false)) { sumExpected = np.CumSum(Replace(a, 0), axis); prodExpected = np.CumProd(Replace(a, 1), axis); }
            Same(sumExpected, np.NanCumSum(a, axis), $"nancumsum {dtype} axis={axis}");
            Same(prodExpected, np.NanCumProd(a, axis), $"nancumprod {dtype} axis={axis}");
        }
    }

    [Fact]
    public void IntegersAndBoolGoToThePlainFunctionsAndComplexIsRefused()
    {
        var ints = KernelPathTests.Make(DType.Int32, false, 4, 5, 6);
        using (KernelPath.Begin(false))
        {
            Same(np.Sum(ints, [1]), np.NanSum(ints, [1]), "nansum int");
            Same(np.Mean(ints, [2]), np.NanMean(ints, [2]), "nanmean int");
            Same(np.Max(ints), np.NanMax(ints), "nanmax int");
            Same(np.Median(ints, [0]), np.NanMedian(ints, [0]), "nanmedian int");
            Same(np.CumSum(ints, 1), np.NanCumSum(ints, 1), "nancumsum int");
        }
        var z = np.Zeros(new[] { 3 }, DType.Complex128);
        Assert.Throws<NDNotSupportedException>(() => np.NanSum(z));
    }
}
