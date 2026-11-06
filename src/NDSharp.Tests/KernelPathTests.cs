// Copyright (c) 2026 Marco Parenzan
// Licensed under the MIT License. See the LICENSE file in the project root.
using NDSharp;
using Xunit;

namespace NDSharp.Tests;

/// <summary>sum, prod, max, min, mean, var, std, cumsum and cumprod run on the MathemaSharp kernels. The general engine is the oracle: for every dtype,
/// layout, axes combination and option, the kernel result must have the same dtype, shape and bits. Counts are per execution context, so other tests
/// running in parallel do not disturb them.</summary>
public class KernelPathTests
{
    internal static readonly DType[] Dtypes = [DType.Int8, DType.UInt8, DType.Int16, DType.UInt16, DType.Int32, DType.UInt32, DType.Int64, DType.UInt64, DType.Float32, DType.Float64, DType.Bool];

    public static IEnumerable<object[]> DtypeCases() => Dtypes.Select(d => new object[] { d });

    /// <summary>Deterministic data of the given dtype; floats get a few special values.</summary>
    internal static NDArray Make(DType dtype, bool specials, params int[] shape)
    {
        int n = 1; foreach (var s in shape) n *= s;
        var d = new double[n];
        uint state = 12345;
        for (int i = 0; i < n; i++)
        {
            state = state * 1664525u + 1013904223u;
            d[i] = Math.Round(Math.Sin((i + 1) * 0.37) * 60.0 + ((state >> 24) / 8.0), 3);
        }
        if (specials && n > 12) { d[3] = double.NaN; d[7] = double.PositiveInfinity; d[11] = -0.0; }
        return NDArray.FromArray(d, shape).AsType(dtype);
    }

    internal static void SameBitsPublic(NDArray expected, NDArray actual, string what) => SameBits(expected, actual, what);

    static void SameBits(NDArray expected, NDArray actual, string what)
    {
        Assert.True(expected.DType == actual.DType, $"{what}: dtype {expected.DType} vs {actual.DType}");
        Assert.True(expected.Shape.SequenceEqual(actual.Shape), $"{what}: shape [{string.Join(",", expected.Shape)}] vs [{string.Join(",", actual.Shape)}]");
        var e = expected.Copy().Buffer; var g = actual.Copy().Buffer;
        Assert.Equal(e.Length, g.Length);
        for (int i = 0; i < e.Length; i++)
        {
            object x = e.GetValue(i)!, y = g.GetValue(i)!;
            bool same = x switch
            {
                double dx => BitConverter.DoubleToInt64Bits(dx) == BitConverter.DoubleToInt64Bits((double)y) || (double.IsNaN(dx) && double.IsNaN((double)y)),
                float fx => BitConverter.SingleToInt32Bits(fx) == BitConverter.SingleToInt32Bits((float)y) || (float.IsNaN(fx) && float.IsNaN((float)y)),
                _ => x.Equals(y),
            };
            Assert.True(same, $"{what}: element {i}: {x} vs {y}");
        }
    }

    internal static readonly int[]?[] AxisSets = [null, [0], [1], [2], [0, 1], [1, 2], [0, 2], [0, 1, 2], [-1]];

    /// <summary>Runs the call with the engine and with the kernels and compares; returns whether the kernels answered.</summary>
    internal static bool Compare(string what, Func<NDArray> call)
    {
        NDArray expected; Exception? expectedError = null;
        using (var off = KernelPath.Begin(false))
        {
            try { expected = call(); } catch (Exception e) { expected = null!; expectedError = e; }
            Assert.Equal(0, off.Calls);
        }
        using var on = KernelPath.Begin(true);
        if (expectedError is not null)
        {
            var actual = Assert.ThrowsAny<Exception>(call);
            Assert.Equal(expectedError.GetType(), actual.GetType());
            Assert.Equal(expectedError.Message, actual.Message);
            return false;
        }
        var got = call();
        SameBits(expected, got, what);
        return on.Calls > 0;
    }

    [Theory, MemberData(nameof(DtypeCases))]
    public void ReductionsMatchTheEngineOnEveryAxesCombination(DType dtype)
    {
        bool isFloat = dtype is DType.Float32 or DType.Float64;
        foreach (var specials in isFloat ? new[] { false, true } : [false])
        {
            var a = Make(dtype, specials, 4, 5, 6);
            foreach (var axes in AxisSets)
            foreach (var keep in new[] { false, true })
            {
                string at = $"{dtype} specials={specials} axes=[{(axes is null ? "all" : string.Join(",", axes))}] keep={keep}";
                Assert.True(Compare("sum " + at, () => np.Sum(a, axes, keep)), "sum not on kernels " + at);
                Assert.True(Compare("prod " + at, () => np.Prod(a, axes, keep)), "prod not on kernels " + at);
                Assert.True(Compare("max " + at, () => np.Max(a, axes, keep)), "max not on kernels " + at);
                Assert.True(Compare("min " + at, () => np.Min(a, axes, keep)), "min not on kernels " + at);
                Assert.True(Compare("mean " + at, () => np.Mean(a, axes, keep)), "mean not on kernels " + at);
                foreach (var ddof in new[] { 0, 1 })
                {
                    Assert.True(Compare($"var ddof={ddof} " + at, () => np.Var(a, axes, keep, ddof)), "var not on kernels " + at);
                    Assert.True(Compare($"std ddof={ddof} " + at, () => np.Std(a, axes, keep, ddof)), "std not on kernels " + at);
                }
            }
        }
    }

    [Theory, MemberData(nameof(DtypeCases))]
    public void ScansMatchTheEngine(DType dtype)
    {
        var a = Make(dtype, true, 4, 5, 6);
        foreach (int? axis in new int?[] { null, 0, 1, 2, -1 })
        {
            Assert.True(Compare($"cumsum {dtype} axis={axis}", () => np.CumSum(a, axis)));
            Assert.True(Compare($"cumprod {dtype} axis={axis}", () => np.CumProd(a, axis)));
        }
    }

    [Theory, MemberData(nameof(DtypeCases))]
    public void EveryLayoutOfTheInputGivesTheSameBits(DType dtype)
    {
        var baseArray = Make(dtype, true, 4, 5, 6);
        var views = new (string, NDArray)[]
        {
            ("transposed", np.Transpose(baseArray)),
            ("swapped", np.SwapAxes(baseArray, 0, 2)),
            ("flipped", np.Flip(baseArray, 1)),
            ("moved", np.MoveAxis(baseArray, 0, 2)),
            ("broadcast", np.BroadcastTo(Make(dtype, false, 1, 5, 6), [4, 5, 6])),
        };
        foreach (var (name, v) in views)
            foreach (var axes in AxisSets)
            {
                if (axes is { } ax && ax.Any(x => x >= 3)) continue;
                string at = $"{name} {dtype} axes=[{(axes is null ? "all" : string.Join(",", axes))}]";
                Assert.True(Compare("sum " + at, () => np.Sum(v, axes)), at);
                Assert.True(Compare("max " + at, () => np.Max(v, axes)), at);
                Assert.True(Compare("mean " + at, () => np.Mean(v, axes)), at);
                Assert.True(Compare("var " + at, () => np.Var(v, axes, false, 1)), at);
                Assert.True(Compare("cumsum " + at, () => np.CumSum(v, axes is { Length: 1 } one ? one[0] : null)), at);
            }
    }

    [Fact]
    public void SumOfNegativeZerosKeepsItsSignWhateverTheLayout()
    {
        var z = NDArray.FromArray(new[] { -0.0, -0.0, -0.0, -0.0, -0.0, -0.0 }, 2, 3);
        Assert.True(Compare("contiguous", () => np.Sum(z, [1])));
        Assert.True(Compare("axis 0", () => np.Sum(z, [0])));
        Assert.True(Compare("transposed", () => np.Sum(np.Transpose(z), [0])));
    }

    [Fact]
    public void LargeAndLongLanesUseTheSameSummationOrder()
    {
        foreach (var n in new[] { 7, 8, 9, 127, 128, 129, 1000, 5000 })
        {
            var a = Make(DType.Float32, false, n);
            Assert.True(Compare($"float32 sum n={n}", () => np.Sum(a)));
            Assert.True(Compare($"float32 mean n={n}", () => np.Mean(a)));
            Assert.True(Compare($"float32 var n={n}", () => np.Var(a)));
            var b = Make(DType.Float64, false, 3, n);
            Assert.True(Compare($"float64 axis 1 n={n}", () => np.Sum(b, [1])));
            Assert.True(Compare($"float64 axis 0 n={n}", () => np.Sum(b, [0])));
        }
    }

    [Fact]
    public void EmptyAndDegenerateInputs()
    {
        var empty = NDArray.FromArray(Array.Empty<double>(), 0);
        var emptyRows = NDArray.FromArray(Array.Empty<double>(), 0, 3);
        var one = NDArray.FromArray(new[] { 4.0 }, 1);
        Compare("sum empty", () => np.Sum(empty));
        Compare("prod empty", () => np.Prod(empty));
        Compare("mean empty", () => np.Mean(empty));
        Compare("var empty", () => np.Var(empty));
        Compare("max empty", () => np.Max(empty));            // both throw the same error
        Compare("min empty rows", () => np.Min(emptyRows, [0]));
        Compare("sum empty rows axis 1", () => np.Sum(emptyRows, [1]));
        Compare("sum empty rows axis 0", () => np.Sum(emptyRows, [0]));
        Compare("cumsum empty", () => np.CumSum(empty));
        Compare("var one ddof 1", () => np.Var(one, null, false, 1));
        Compare("var one ddof 2", () => np.Var(one, null, false, 2));
        var two = NDArray.FromArray(new[] { 1.0, 2.0 }, 2);
        Compare("var two ddof 3", () => np.Var(two, null, false, 3));
        Compare("scalar sum", () => np.Sum(NDArray.Scalar(3.5)));
        Compare("scalar cumsum", () => np.CumSum(NDArray.Scalar(3.5)));
        Compare("scalar max", () => np.Max(NDArray.Scalar(3.5)));
    }

    [Fact]
    public void OptionsDtypeAndErrors()
    {
        var a = Make(DType.Int32, false, 4, 5, 6);
        Assert.True(Compare("sum dtype float64", () => np.Sum(a, [1], false, DType.Float64)));
        Assert.True(Compare("sum dtype float32", () => np.Sum(a, [1], false, DType.Float32)));
        Assert.True(Compare("mean dtype float32", () => np.Mean(a, [2], false, DType.Float32)));
        Assert.True(Compare("cumsum dtype", () => np.CumSum(a, 1, DType.Float64)));
        Compare("bad axis", () => np.Sum(a, [3]));
        Compare("duplicate axis", () => np.Sum(a, [1, 1]));
        Compare("bad cum axis", () => np.CumSum(a, 5));
    }

    [Fact]
    public void Float16AndComplexStayOnTheGeneralEngine()
    {
        var h = Make(DType.Float64, false, 4, 5).AsType(DType.Float16);
        using var scope = KernelPath.Begin(true);
        _ = np.Sum(h, [0]); _ = np.Max(h); _ = np.Mean(h); _ = np.CumSum(h, 0);
        Assert.Equal(0, scope.Calls);
    }

    [Fact]
    public void SwitchOffUsesTheEngineAndScopesNest()
    {
        var a = Make(DType.Float64, false, 10);
        using var outer = KernelPath.Begin(true);
        _ = np.Sum(a);
        using (var inner = KernelPath.Begin(false)) { _ = np.Sum(a); Assert.Equal(0, inner.Calls); }
        _ = np.Sum(a);
        Assert.Equal(2, outer.Calls);
    }
}
