using System.Numerics;
using NDSharp.Mathema;
using NDSharp;

namespace NDSharp.Mathema.Tests;

public class NativeConversionTests
{
    [Fact]
    public void TypedStorageRetainsDecimal128Increment()
    {
        var delta = Decimal128.Parse("1e-25", System.Globalization.CultureInfo.InvariantCulture);
        Decimal128[] values = [Decimal128.One, delta];
        var array = values.ToTypedNDArray();
        Assert.Equal(values, array.ToVector());
        Assert.Equal(delta, array.Sum() - Decimal128.One);
        Assert.Equal(delta, array.Dot(new NDArray<Decimal128>(new Decimal128[] { Decimal128.One, Decimal128.One }, 2)) - Decimal128.One);
        var matrix = new NDArray<Decimal128>(values, 1, 2);
        var column = new NDArray<Decimal128>(new Decimal128[] { Decimal128.One, Decimal128.One }, 2, 1);
        Assert.Equal(delta, matrix.MatMul(column)[0, 0] - Decimal128.One);
    }

    [Fact]
    public void TypedStorageSupportsAllSixFloatingFormats()
    {
        CheckTyped<Half>(); CheckTyped<float>(); CheckTyped<double>();
        CheckTyped<Decimal32>(); CheckTyped<Decimal64>(); CheckTyped<Decimal128>();
    }

    private static void CheckTyped<T>() where T : IFloatingPointIeee754<T>
    {
        T[] values = [T.One, T.CreateChecked(2), T.CreateChecked(3)];
        var array = values.ToTypedNDArray();
        Assert.Equal(values, array.ToVector());
        Assert.Equal(T.CreateChecked(6), array.Sum());
        Assert.Equal(T.CreateChecked(12), array.Add(array).Sum());
        Assert.Equal(T.CreateChecked(14), array.Dot(array));
        var matrix = new NDArray<T>(values, 1, 3);
        Assert.Equal(T.CreateChecked(14), matrix.MatMul(matrix.Transpose())[0, 0]);
        var reversed = array.Slice(0, 2, 3, -1);
        Assert.Equal(new T[] { T.CreateChecked(3), T.CreateChecked(2), T.One }, reversed.ToVector());
        var broadcast = reversed.Add(new NDArray<T>(new T[] { T.One }));
        Assert.Equal(new T[] { T.CreateChecked(4), T.CreateChecked(3), T.CreateChecked(2) }, broadcast.ToArray());
    }

    [Fact]
    public void Int64DoesNotPassThroughDouble()
    {
        long[] values = [9_007_199_254_740_993, long.MaxValue];
        var array = NDArray.FromArray(values);
        Assert.Equal(values, array.ToVector<long>());
        Assert.Equal(values.Select(x => (decimal)x), array.ToVector<decimal>());
        Assert.Equal(values.Select(x => Decimal128.CreateChecked(x)), array.ToVector<Decimal128>());
    }

    [Fact]
    public void UInt64MaximumIsPreserved()
    {
        var array = NDArray.FromArray(new ulong[] { ulong.MaxValue });
        Assert.Equal(ulong.MaxValue, array.ToVector<ulong>()[0]);
        Assert.Equal((decimal)ulong.MaxValue, array.ToVector<decimal>()[0]);
    }

    [Fact]
    public void MatrixCopiesNativeIntegerValues()
    {
        var source = new long[,] { { 9_007_199_254_740_993, 2 }, { 3, 4 } };
        var converted = NDArray.FromMultiDim(source).ToMatrix<Decimal128>();
        Assert.Equal(Decimal128.CreateChecked(source[0, 0]), converted[0, 0]);
        Assert.Equal(Decimal128.CreateChecked(4), converted[1, 1]);
    }

    [Fact]
    public void CheckedIntegerOverflowIsReported()
    {
        Assert.Throws<OverflowException>(() => NDArray.FromArray(new long[] { long.MaxValue }).ToVector<int>());
    }
}
