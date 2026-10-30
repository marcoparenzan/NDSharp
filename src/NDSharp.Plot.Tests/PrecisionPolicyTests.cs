namespace NDSharp.Plot.Tests;

/// <summary>
/// Pins the display-conversion policy (R06): plotting works on <c>double</c>; the conversion from NDArray storage is exact
/// for float32 and for integers up to 2^53, and rounds beyond that. Stored data are never modified.
/// </summary>
public class PrecisionPolicyTests
{
    const long TwoPow53 = 1L << 53;

    [Fact]
    public void Int64BeyondTwoToThe53RoundsWhenConvertedForDisplay()
    {
        var stored = new NDArray(DType.Int64, new long[] { TwoPow53, TwoPow53 + 1, TwoPow53 + 2 }, new[] { 3 });
        var shown = stored.ToArray<double>();
        Assert.Equal((double)TwoPow53, shown[0]);
        Assert.Equal((double)TwoPow53, shown[1]);              // 2^53 + 1 is not representable: ties to even
        Assert.Equal((double)(TwoPow53 + 2), shown[2]);
        Assert.Equal(TwoPow53 + 1, Convert.ToInt64(stored.GetAt(1))); // the stored value is untouched
    }

    [Fact]
    public void IntegersUpToTwoToThe53AreExact()
    {
        var stored = new NDArray(DType.Int64, new long[] { TwoPow53 - 1, -(TwoPow53 - 1) }, new[] { 2 });
        var shown = stored.ToArray<double>();
        Assert.Equal(TwoPow53 - 1, (long)shown[0]);
        Assert.Equal(-(TwoPow53 - 1), (long)shown[1]);
    }

    [Fact]
    public void Float32WidensExactlyToDouble()
    {
        var stored = new NDArray(DType.Float32, new float[] { 0.1f }, new[] { 1 });
        Assert.Equal((double)0.1f, stored.ToArray<double>()[0]);
        Assert.NotEqual(0.1, stored.ToArray<double>()[0]);     // widening does not recover the decimal 0.1
    }

    [Fact]
    public void SvgLabelsUseAFixedShortFormatNotFullPrecision()
    {
        Assert.Equal("1.2e6", PlotSharp.Charts.Num(1234567.891));
        Assert.Equal("0", PlotSharp.Charts.Num(0));
    }
}
