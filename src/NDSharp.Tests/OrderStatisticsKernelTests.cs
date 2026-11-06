// Copyright (c) 2026 Marco Parenzan
// Licensed under the MIT License. See the LICENSE file in the project root.
using NDSharp;
using Xunit;

namespace NDSharp.Tests;

/// <summary>median, percentile and quantile run on the MathemaSharp kernels. For the 'linear' method and for median the general engine is the oracle
/// (same dtype, shape and bits); the other methods are checked against values worked out by hand.</summary>
public class OrderStatisticsKernelTests
{
    public static IEnumerable<object[]> DtypeCases() => KernelPathTests.Dtypes.Select(d => new object[] { d });

    static NDArray Q(params double[] q) => NDArray.FromArray(q, q.Length);

    [Theory, MemberData(nameof(DtypeCases))]
    public void MedianMatchesTheEngine(DType dtype)
    {
        bool isFloat = dtype is DType.Float16 or DType.Float32 or DType.Float64;
        foreach (var specials in isFloat ? new[] { false, true } : [false])
        {
            var a = KernelPathTests.Make(dtype, specials, 4, 5, 6);
            foreach (var axes in KernelPathTests.AxisSets)
            foreach (var keep in new[] { false, true })
            {
                string at = $"{dtype} specials={specials} axes=[{(axes is null ? "all" : string.Join(",", axes))}] keep={keep}";
                Assert.True(KernelPathTests.Compare("median " + at, () => np.Median(a, axes, keep)), "median not on kernels " + at);
            }
        }
    }

    [Theory, MemberData(nameof(DtypeCases))]
    public void LinearPercentileMatchesTheEngine(DType dtype)
    {
        bool isFloat = dtype is DType.Float16 or DType.Float32 or DType.Float64;
        foreach (var specials in isFloat ? new[] { false, true } : [false])
        {
            var a = KernelPathTests.Make(dtype, specials, 4, 5, 6);
            foreach (var axes in KernelPathTests.AxisSets)
            foreach (var keep in new[] { false, true })
            foreach (var isQuantile in new[] { false, true })
            {
                string at = $"{dtype} specials={specials} axes=[{(axes is null ? "all" : string.Join(",", axes))}] keep={keep} quantile={isQuantile}";
                double scale = isQuantile ? 0.01 : 1;
                Assert.True(KernelPathTests.Compare("scalar q " + at, () => np.Percentile(a, NDArray.Scalar(37.5 * scale), axes, keep, isQuantile)), at);
                Assert.True(KernelPathTests.Compare("vector q " + at, () => np.Percentile(a, Q(0, 12.5 * scale, 50 * scale, 99.9 * scale, 100 * scale), axes, keep, isQuantile)), at);
            }
        }
    }

    [Fact]
    public void LayoutsGiveTheSameBits()
    {
        var baseArray = KernelPathTests.Make(DType.Float64, true, 4, 5, 6);
        foreach (var (name, v) in new[] { ("transposed", np.Transpose(baseArray)), ("flipped", np.Flip(baseArray, 1)), ("moved", np.MoveAxis(baseArray, 0, 2)) })
            foreach (var axes in new int[]?[] { null, [0], [1], [0, 2], [1, 2] })
            {
                Assert.True(KernelPathTests.Compare($"median {name} axes=[{(axes is null ? "all" : string.Join(",", axes))}]", () => np.Median(v, axes)), name);
                Assert.True(KernelPathTests.Compare($"percentile {name} axes=[{(axes is null ? "all" : string.Join(",", axes))}]", () => np.Percentile(v, Q(10, 90), axes)), name);
            }
    }

    [Fact]
    public void MedianOfTwoHugeValuesOverflowsLikeNumpy()
    {
        var a = NDArray.FromArray(new[] { double.MaxValue, double.MaxValue }, 2);
        Assert.True(KernelPathTests.Compare("median realmax", () => np.Median(a)));
        using var _ = KernelPath.Begin();
        Assert.True(double.IsPositiveInfinity(np.Median(a).Data<double>()[0]));   // numpy: (a + b) / 2 overflows
    }

    [Fact]
    public void EmptyAndErrors()
    {
        KernelPathTests.Compare("median empty", () => np.Median(NDArray.FromArray(Array.Empty<double>(), 0)));
        KernelPathTests.Compare("percentile empty", () => np.Percentile(NDArray.FromArray(Array.Empty<double>(), 0), NDArray.Scalar(50.0)));
        KernelPathTests.Compare("percentile out of range", () => np.Percentile(NDArray.FromArray(new[] { 1.0, 2.0 }, 2), NDArray.Scalar(101.0)));
        KernelPathTests.Compare("quantile out of range", () => np.Percentile(NDArray.FromArray(new[] { 1.0, 2.0 }, 2), NDArray.Scalar(1.5), null, false, true));
        Assert.Throws<NDValueException>(() => np.Percentile(NDArray.FromArray(new[] { 1.0, 2.0 }, 2), NDArray.Scalar(50.0), null, false, false, "bogus"));
    }

    // [1,2,3,4] at q = 0.25, worked out by hand from the definitions (see the kernels' QuantileTests).
    [Theory]
    [InlineData("inverted_cdf", 1.0)]
    [InlineData("averaged_inverted_cdf", 1.5)]
    [InlineData("closest_observation", 1.0)]
    [InlineData("interpolated_inverted_cdf", 1.0)]
    [InlineData("hazen", 1.5)]
    [InlineData("weibull", 1.25)]
    [InlineData("linear", 1.75)]
    [InlineData("median_unbiased", 1.4166666666666667)]
    [InlineData("normal_unbiased", 1.4375)]
    [InlineData("lower", 1.0)]
    [InlineData("higher", 2.0)]
    [InlineData("nearest", 2.0)]
    [InlineData("midpoint", 1.5)]
    public void NumpyMethodNames(string method, double expected)
    {
        var a = NDArray.FromArray(new[] { 3.0, 1.0, 4.0, 2.0 }, 4);
        var r = np.Percentile(a, NDArray.Scalar(0.25), null, false, true, method);
        Assert.Equal(expected, r.Data<double>()[0], 12);
    }

    [Fact]
    public void MethodsAlongAxesKeepTheProbabilityAxisFirst()
    {
        var a = NDArray.FromArray(new[] { 1.0, 10.0, 2.0, 20.0, 3.0, 30.0 }, 3, 2);
        var r = np.Percentile(a, Q(0, 50, 100), new[] { 0 }, false, false, "hazen");
        Assert.Equal(new[] { 3, 2 }, r.Shape);
        Assert.Equal(new[] { 1.0, 10.0, 2.0, 20.0, 3.0, 30.0 }, r.Copy().Data<double>());
    }
}
