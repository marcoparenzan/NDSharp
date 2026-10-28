using System.Numerics;

namespace NDSharp.Tests;

public class NormTests
{
    static NDArray Cx(params Complex[] v) => new(DType.Complex128, v, new[] { v.Length });

    [Fact]
    public void ComplexVectorNormUsesTheModuli()
    {
        // |3+4i| = 5, not the real part and not the complex square root of z^2
        Assert.Equal(5.0, Convert.ToDouble(np.Norm(Cx(new Complex(3, 4), Complex.Zero)).GetAt(0)), 12);
        Assert.Equal(Math.Sqrt(2 * 2 + 3 * 3 + 1), Convert.ToDouble(np.Norm(Cx(new Complex(2, 3), new Complex(0, 1))).GetAt(0)), 12);
    }

    [Fact]
    public void ComplexNormAlongAxesMatchesNumpy()
    {
        // numpy: m = [[3+4j, 1j], [0j, 1+0j]]  ->  axis0 [5, sqrt(2)], axis1 [sqrt(26), 1], fro sqrt(27)
        var m = new NDArray(DType.Complex128, new[] { new Complex(3, 4), new Complex(0, 1), Complex.Zero, Complex.One }, new[] { 2, 2 });
        var a0 = np.Norm(m, new[] { 0 }).AsType(DType.Float64).ToArray<double>();
        var a1 = np.Norm(m, new[] { 1 }).AsType(DType.Float64).ToArray<double>();
        Assert.Equal(new[] { 5.0, Math.Round(Math.Sqrt(2), 12) }, a0.Select(x => Math.Round(x, 12)).ToArray());
        Assert.Equal(new[] { Math.Round(Math.Sqrt(26), 12), 1.0 }, a1.Select(x => Math.Round(x, 12)).ToArray());
        Assert.Equal(Math.Sqrt(27), Convert.ToDouble(np.Norm(m).GetAt(0)), 12);
    }

    [Fact]
    public void ComplexNormResultIsReal()
        => Assert.False(np.Norm(Cx(new Complex(1, 1))).DType.IsComplex());

    [Fact]
    public void ComplexOneAndInfNormsUseModuli()
    {
        var v = Cx(new Complex(3, 4), new Complex(0, -2));
        Assert.Equal(7.0, Convert.ToDouble(np.Norm(v, 1.0).GetAt(0)), 12);
        Assert.Equal(5.0, Convert.ToDouble(np.Norm(v, double.PositiveInfinity).GetAt(0)), 12);
    }
}

