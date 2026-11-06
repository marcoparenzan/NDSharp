// Copyright (c) 2026 Marco Parenzan
// Licensed under the MIT License. See the LICENSE file in the project root.
using NDSharp;
using Xunit;

namespace NDSharp.Tests;

/// <summary>matmul/dot, cov and corrcoef run on the MathemaSharp kernels. The general loops of the engine (kernel path off) are the oracle: same dtype, shape and bits.</summary>
public class ProductsKernelTests
{
    static readonly DType[] ProductTypes = [DType.Int8, DType.UInt8, DType.Int16, DType.Int32, DType.UInt32, DType.Int64, DType.UInt64, DType.Float16, DType.Float32, DType.Float64, DType.Complex64, DType.Complex128];

    public static IEnumerable<object[]> ProductCases() => ProductTypes.Select(d => new object[] { d });

    static bool IsFloating(DType d) => d is DType.Float16 or DType.Float32 or DType.Float64 or DType.Complex64 or DType.Complex128;

    [Theory, MemberData(nameof(ProductCases))]
    public void MatMulMatchesTheEngine(DType dtype)
    {
        bool sp = IsFloating(dtype);
        var a = KernelPathTests.Make(dtype, sp, 3, 4);
        var b = KernelPathTests.Make(dtype, sp, 4, 5);
        var a3 = KernelPathTests.Make(dtype, sp, 2, 3, 4);
        var b3 = KernelPathTests.Make(dtype, sp, 2, 4, 5);
        var v4 = KernelPathTests.Make(dtype, sp, 4);
        Assert.True(KernelPathTests.Compare($"2d {dtype}", () => np.MatMul(a, b)));
        Assert.True(KernelPathTests.Compare($"batched {dtype}", () => np.MatMul(a3, b3)));
        Assert.True(KernelPathTests.Compare($"broadcast batch {dtype}", () => np.MatMul(KernelPathTests.Make(dtype, sp, 1, 3, 4), KernelPathTests.Make(dtype, sp, 5, 4, 2))));
        Assert.True(KernelPathTests.Compare($"2d x batch {dtype}", () => np.MatMul(a, b3)));
        Assert.True(KernelPathTests.Compare($"vector . vector {dtype}", () => np.MatMul(v4, v4)));
        Assert.True(KernelPathTests.Compare($"matrix . vector {dtype}", () => np.MatMul(a, v4)));
        Assert.True(KernelPathTests.Compare($"vector . matrix {dtype}", () => np.MatMul(v4, b)));
        Assert.True(KernelPathTests.Compare($"dot {dtype}", () => np.Dot(a, b)));
        Assert.True(KernelPathTests.Compare($"transposed a {dtype}", () => np.MatMul(np.Transpose(KernelPathTests.Make(dtype, sp, 4, 3)), b)));
        Assert.True(KernelPathTests.Compare($"transposed b {dtype}", () => np.MatMul(a, np.Transpose(KernelPathTests.Make(dtype, sp, 5, 4)))));
        Assert.True(KernelPathTests.Compare($"flipped {dtype}", () => np.MatMul(np.Flip(a, 1), np.Flip(b, 0))));
        Assert.True(KernelPathTests.Compare($"batch of transposed {dtype}", () => np.MatMul(np.SwapAxes(KernelPathTests.Make(dtype, sp, 2, 4, 3), 1, 2), b3)));
    }

    [Fact]
    public void MatMulEmptyAndErrors()
    {
        var empty = np.Zeros(new[] { 3, 0 }, DType.Float64);
        var emptyB = np.Zeros(new[] { 0, 2 }, DType.Float64);
        Assert.True(KernelPathTests.Compare("k = 0", () => np.MatMul(empty, emptyB)));
        Assert.True(KernelPathTests.Compare("m = 0", () => np.MatMul(np.Zeros(new[] { 0, 4 }, DType.Float64), KernelPathTests.Make(DType.Float64, false, 4, 2))));
        KernelPathTests.Compare("mismatch", () => np.MatMul(KernelPathTests.Make(DType.Float64, false, 3, 4), KernelPathTests.Make(DType.Float64, false, 5, 2)));
        KernelPathTests.Compare("scalar", () => np.MatMul(NDArray.Scalar(2.0), KernelPathTests.Make(DType.Float64, false, 2)));
    }

    [Fact]
    public void MatMulKnownProduct()
    {
        using var _ = KernelPath.Begin();
        var a = NDArray.FromArray(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }, 2, 3);
        var b = NDArray.FromArray(new[] { 7.0, 8.0, 9.0, 10.0, 11.0, 12.0 }, 3, 2);
        Assert.Equal(new[] { 58.0, 64.0, 139.0, 154.0 }, np.MatMul(a, b).Copy().Data<double>());
    }

    static NDArray Series(DType dtype, bool specials, int variables, int observations)
        => KernelPathTests.Make(dtype, specials, variables, observations);

    [Theory]
    [InlineData(DType.Float64)]
    [InlineData(DType.Float32)]
    [InlineData(DType.Float16)]
    [InlineData(DType.Int32)]
    [InlineData(DType.UInt8)]
    [InlineData(DType.Bool)]
    public void CovMatchesTheEngine(DType dtype)
    {
        bool sp = IsFloating(dtype);
        var m = Series(dtype, sp, 3, 20);
        var y = Series(dtype, false, 2, 20);
        foreach (var rowvar in new[] { true, false })
        foreach (var bias in new[] { false, true })
        foreach (int? ddof in new int?[] { null, 0, 1, 2, 19, 20, 25 })
        {
            string at = $"{dtype} rowvar={rowvar} bias={bias} ddof={ddof}";
            var data = rowvar ? m : np.Transpose(m);
            Assert.True(KernelPathTests.Compare("cov " + at, () => np.Cov(data, null, rowvar, bias, ddof)), at);
            var other = rowvar ? y : np.Transpose(y);
            Assert.True(KernelPathTests.Compare("cov y " + at, () => np.Cov(data, other, rowvar, bias, ddof)), at);
        }
        Assert.True(KernelPathTests.Compare($"1-D {dtype}", () => np.Cov(KernelPathTests.Make(dtype, sp, 20))));
        Assert.True(KernelPathTests.Compare($"two 1-D {dtype}", () => np.Cov(KernelPathTests.Make(dtype, sp, 20), KernelPathTests.Make(dtype, false, 20))));
        Assert.True(KernelPathTests.Compare($"strided {dtype}", () => np.Cov(np.Flip(m, 1))));
    }

    [Fact]
    public void CovDegenerateInputs()
    {
        KernelPathTests.Compare("one observation", () => np.Cov(NDArray.FromArray(new[] { 1.0, 2.0, 3.0 }, 3, 1)));
        KernelPathTests.Compare("one observation, ddof 0", () => np.Cov(NDArray.FromArray(new[] { 1.0, 2.0, 3.0 }, 3, 1), null, true, true));
        KernelPathTests.Compare("no observations", () => np.Cov(np.Zeros(new[] { 2, 0 }, DType.Float64)));
        KernelPathTests.Compare("scalar", () => np.Cov(NDArray.Scalar(3.0)));
    }

    [Theory]
    [InlineData(DType.Float64)]
    [InlineData(DType.Float32)]
    [InlineData(DType.Int32)]
    public void CorrCoefMatchesTheEngine(DType dtype)
    {
        bool sp = dtype != DType.Int32;
        var m = Series(dtype, sp, 4, 25);
        var y = Series(dtype, false, 2, 25);
        foreach (var rowvar in new[] { true, false })
        {
            var data = rowvar ? m : np.Transpose(m);
            Assert.True(KernelPathTests.Compare($"corrcoef {dtype} rowvar={rowvar}", () => np.CorrCoef(data, null, rowvar)));
            var other = rowvar ? y : np.Transpose(y);
            Assert.True(KernelPathTests.Compare($"corrcoef y {dtype} rowvar={rowvar}", () => np.CorrCoef(data, other, rowvar)));
        }
        // a constant row has no variance: NaN off the diagonal, as numpy
        var constant = NDArray.FromArray(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 7.0, 7.0, 7.0, 7.0 }, 2, 5);
        Assert.True(KernelPathTests.Compare("constant row", () => np.CorrCoef(constant)));
        KernelPathTests.Compare("single variable", () => np.CorrCoef(NDArray.FromArray(new[] { 1.0, 2.0, 4.0 }, 3)));
    }

    [Fact]
    public void CorrCoefIsClippedAndHasKnownValues()
    {
        using var _ = KernelPath.Begin();
        var x = NDArray.FromArray(new[] { 1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0 }, 2, 4);   // perfectly correlated
        Assert.Equal(new[] { 1.0, 1.0, 1.0, 1.0 }, np.CorrCoef(x).Copy().Data<double>());
        var anti = NDArray.FromArray(new[] { 1.0, 2.0, 3.0, 4.0, 8.0, 6.0, 4.0, 2.0 }, 2, 4);
        Assert.Equal(-1.0, np.CorrCoef(anti).Copy().Data<double>()[1]);
    }
}
