// Copyright (c) 2026 Marco Parenzan
// Licensed under the MIT License. See the LICENSE file in the project root.
using Xunit;

namespace NDSharp.Tests;

public class ArrayViewTests
{
    [Fact]
    public void RoundTripSharesStorageAndPreservesSignedZeroBits()
    {
        double[] data = [-0.0, double.NaN, double.PositiveInfinity, double.Epsilon, 4, 5];
        var original = new NDArray(DType.Float64, data, [2, 3]);
        var view = original.AsArrayView<double>();
        var result = NDArray.FromArrayView(view, original.DType);
        Assert.Same(data, result.Buffer);
        for (int i = 0; i < data.Length; i++)
            Assert.Equal(BitConverter.DoubleToInt64Bits(data[i]), BitConverter.DoubleToInt64Bits(result.Data<double>()[i]));
        view[1, 2] = 17;
        Assert.Equal(17d, original.GetDouble(5));
    }

    [Fact]
    public void NegativeAndBroadcastStridesShareWrites()
    {
        int[] data = [1, 2, 3];
        var view = new ArrayView<int>(data, [2, 3], [0, -1], 2);
        Assert.Equal(3, view[0, 0]);
        view[1, 2] = 8;
        Assert.Equal(8, view[0, 2]);
        var roundTrip = NDArray.FromArrayView(view, DType.Int32);
        Assert.Equal(new[] { 0, -1 }, roundTrip.Strides);
        Assert.Equal(2, roundTrip.Offset);
    }

    [Fact]
    public void ColumnMajorAndMetadataSnapshotsArePreserved()
    {
        int[] shape = [2, 3], strides = [1, 2];
        var view = new ArrayView<int>([0, 1, 2, 3, 4, 5], shape, strides);
        shape[0] = 99;
        strides[0] = 99;
        Assert.Equal(5, view[1, 2]);
        Assert.Equal(2, view.Shape[0]);
        Assert.Equal(1, view.Strides[0]);
    }

    [Fact]
    public void ScalarsAndEmptyArraysHaveDefinedBounds()
    {
        var scalar = new ArrayView<int>([42], [], []);
        Assert.Equal(1, scalar.Size);
        Assert.Equal(42, scalar[Array.Empty<int>()]);
        var empty = new ArrayView<int>([], [0, int.MaxValue], [int.MinValue, int.MaxValue]);
        Assert.Equal(0, empty.Size);
        Assert.Throws<ArgumentOutOfRangeException>(() => empty[0, 0]);
    }

    [Fact]
    public void InvalidMetadataAndIndicesAreRejected()
    {
        Assert.Throws<ArgumentException>(() => new ArrayView<int>([1], [1], []));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ArrayView<int>([1], [-1], [1]));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ArrayView<int>([1], [2], [1]));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ArrayView<int>([1], [2], [-1]));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ArrayView<int>([1], [int.MaxValue, 2], [0, 0]));
        var view = new ArrayView<int>([1], [1], [1]);
        Assert.Throws<ArgumentException>(() => view[0, 0]);
        Assert.Throws<ArgumentOutOfRangeException>(() => view[-1]);
    }

    [Fact]
    public void SignednessAndDtypeMustMatchExactly()
    {
        var array = new NDArray(DType.UInt32, new uint[] { 1 }, [1]);
        Assert.Throws<NDTypeException>(() => array.AsArrayView<int>());
        Assert.Throws<NDTypeException>(() => NDArray.FromArrayView(array.AsArrayView<uint>(), DType.Int32));
        Assert.Equal(DType.UInt32, NDArray.FromArrayView(array.AsArrayView<uint>(), DType.UInt32).DType);
    }
}
