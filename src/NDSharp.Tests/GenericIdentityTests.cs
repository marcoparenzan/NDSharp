using NDSharp;
using Xunit;

/// <summary>Integers keep every bit through sums and products, and empty reductions give the neutral element of their dtype (these used to test the private identity helper of the former engine).</summary>
public class GenericIdentityTests
{
    [Fact] public void Int64SumRetainsLowBits() => Assert.Equal(9007199254740993L, np.Sum(NDArray.FromArray(new long[] { 9007199254740992L, 1 }, 2)).Copy().Data<long>()[0]);
    [Fact] public void UInt64SumRetainsMaximum() => Assert.Equal(ulong.MaxValue, np.Sum(NDArray.FromArray(new ulong[] { ulong.MaxValue, 0 }, 2)).Copy().Data<ulong>()[0]);
    [Fact] public void BoolSumCountsInInt64() { var r = np.Sum(NDArray.FromArray(new[] { true, false, true }, 3)); Assert.Equal(DType.Int64, r.DType); Assert.Equal(2L, r.Copy().Data<long>()[0]); }
    [Fact] public void EmptySumAndProductGiveZeroAndOne()
    {
        Assert.Equal(0L, np.Sum(NDArray.FromArray(Array.Empty<long>(), 0)).Copy().Data<long>()[0]);
        Assert.Equal(1L, np.Prod(NDArray.FromArray(Array.Empty<long>(), 0)).Copy().Data<long>()[0]);
        Assert.Equal(1.0, np.Prod(NDArray.FromArray(Array.Empty<double>(), 0)).Copy().Data<double>()[0]);
    }
}
