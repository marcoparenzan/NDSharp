using System.Numerics;
using NDSharp;
using NDSharp.Mathema;

namespace NDSharp.Mathema.Tests;

public class NDArrayBridgeTests
{
    [Fact] public void VectorRoundTripSharesBuffer()
    {
        double[] values = [1, 2, 3];
        var array = values.ToNDArray();
        Assert.Equal([3], array.Shape);
        Assert.Equal(values, array.ToVector());
        values[0] = 42;
        Assert.Equal(42, array.ToVector()[0]);
    }

    [Fact] public void MatrixRoundTripIsRowMajor()
    {
        var m = new double[,] { { 1, 2, 3 }, { 4, 5, 6 } };
        var array = m.ToNDArray();
        Assert.Equal([2, 3], array.Shape);
        Assert.Equal(m, array.ToMatrix());
        // A transposed (strided) view is materialised in C order.
        Assert.Equal(new double[,] { { 1, 4 }, { 2, 5 }, { 3, 6 } }, np.Transpose(array).ToMatrix());
    }

    [Fact] public void IntegerArraysAreCastToDouble() =>
        Assert.Equal([0.0, 1.0, 2.0], np.Arange(0L, 3L).ToVector());

    [Fact] public void ComplexVector()
    {
        Complex<double>[] z = [new(1, 2), new(3, -4)];
        Assert.Equal(z, z.ToNDArray().ToComplexVector());
    }

    [Fact] public void ComplexVectorOfAnyIeeeFormat()
    {
        var array = new Complex<double>[] { new(1.5, -2), new(0.25, 4) }.ToNDArray();
        Assert.Equal(new Complex<float>[] { new(1.5f, -2f), new(0.25f, 4f) }, array.ToComplexVector<float>());
        Assert.Equal(new Complex<double>[] { new(1.5, -2), new(0.25, 4) }, array.ToComplexVector<double>());
        // A complex64 array is stored with double parts too, so the values arrive unchanged.
        Assert.Equal(new Complex<double>[] { new(1.5, -2), new(0.25, 4) }, array.AsType(DType.Complex64).ToComplexVector<double>());
        Assert.Equal(array.ToComplexVector(), new Complex<float>[] { new(1.5f, -2f), new(0.25f, 4f) }.ToNDArray().ToComplexVector());
    }

    [Fact] public void ComplexVectorNeedsAComplexDtypeAndOneDimension()
    {
        Assert.Throws<NotSupportedException>(() => new[] { 1.0, 2.0 }.ToNDArray().ToComplexVector<double>());
        Assert.Throws<ArgumentException>(() => np.Zeros(new[] { 2, 2 }, DType.Complex128).ToComplexVector<double>());
    }

    [Fact] public void SameTypeConversionIsACopyNotAView()
    {
        var array = new[] { 1.0, 2.0, 3.0 }.ToNDArray();
        var copy = array.ToVector<double>();
        copy[0] = 99;
        Assert.Equal([1.0, 2.0, 3.0], array.ToVector<double>());
        Assert.Equal([1L, 2L], np.Arange(1L, 3L).ToVector<long>());
        Assert.Equal([1.0, 2.0], np.Arange(1L, 3L).ToVector<double>());
    }

    [Fact] public void GenericFormatsGoThroughDouble()
    {
        var array = new[] { 0.5, 0.25 }.ToNDArray();
        Assert.Equal(new[] { Decimal128.CreateChecked(0.5), Decimal128.CreateChecked(0.25) }, array.ToVector<Decimal128>());
        Assert.Equal(new[] { 0.5f, 0.25f }, array.ToVector<float>());
        Assert.Equal(new[] { 1.5, 2.0 }, new[] { Decimal64.CreateChecked(1.5), Decimal64.CreateChecked(2) }.ToNDArray().ToVector());
        Assert.Equal(new float[,] { { 1, 2 }, { 3, 4 } }, new double[,] { { 1, 2 }, { 3, 4 } }.ToNDArray().ToMatrix<float>());
    }

    [Fact] public void WrongRankThrows() =>
        Assert.Throws<ArgumentException>(() => new double[,] { { 1 } }.ToNDArray().ToVector());
}
