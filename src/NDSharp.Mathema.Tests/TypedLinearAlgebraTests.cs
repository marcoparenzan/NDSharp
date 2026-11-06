using System.Numerics;
using NDSharp.Mathema;
using NDSharp;

namespace NDSharp.Mathema.Tests;

public class TypedLinearAlgebraTests
{
    [Fact] public void AllSixFormatsUseExistingKernels()
    {
        Check<Half>(); Check<float>(); Check<double>();
        Check<Decimal32>(); Check<Decimal64>(); Check<Decimal128>();
    }
    private static void Check<T>() where T : IFloatingPointIeee754<T>
    {
        var matrix = new T[,] { { T.CreateChecked(2), T.Zero }, { T.Zero, T.CreateChecked(4) } }.ToTypedNDArray();
        var rhs = new T[] { T.CreateChecked(2), T.CreateChecked(8) }.ToTypedNDArray();
        var factor = matrix.FactorizeLu();
        Assert.Equal(new T[] { T.One, T.CreateChecked(2) }, factor.Solve(rhs).ToVector());
        Assert.Equal(new T[] { T.One, T.CreateChecked(2) }, factor.Solve(rhs).ToVector());
        var report = matrix.SolveWithDiagnostics(rhs);
        Assert.Equal(T.CreateChecked(2), report.ConditionInfinityEstimate);
        Assert.False(report.IsIllConditioned);
        var qr = matrix.LeastSquares(rhs).ToVector();
        var tolerance = T.Sqrt(T.BitIncrement(T.One) - T.One) * T.CreateChecked(8);
        Assert.True(T.Abs(qr[0] - T.One) <= tolerance);
        Assert.True(T.Abs(qr[1] - T.CreateChecked(2)) <= tolerance);
    }
    [Fact] public void ViewsAreCopiedInLogicalOrder()
    {
        var source = new double[,] { { 99, 2, 0 }, { 99, 0, 4 } }.ToTypedNDArray();
        var matrix = source.Slice(1, 1, 2).Transpose();
        var rhs = new double[] { 8, 2 }.ToTypedNDArray().Slice(0, 1, 2, -1);
        Assert.Equal(new double[] { 1, 2 }, matrix.FactorizeLu().Solve(rhs).ToVector());
        var copied = matrix.ToMatrix(); copied[0, 0] = 100;
        Assert.Equal(2, source[0, 1]);
    }
    [Fact] public void Decimal128SolutionRetainsSmallIncrement()
    {
        var delta = Decimal128.Parse("1e-25");
        var identity = new Decimal128[,] { { Decimal128.One, Decimal128.Zero }, { Decimal128.Zero, Decimal128.One } }.ToTypedNDArray();
        var rhs = new Decimal128[] { Decimal128.One + delta, Decimal128.One }.ToTypedNDArray();
        Assert.Equal(delta, identity.FactorizeLu().Solve(rhs)[0] - Decimal128.One);
        Assert.Equal(delta, identity.SolveWithDiagnostics(rhs).Solution[0] - Decimal128.One);
        Assert.Equal(delta, identity.LeastSquares(rhs)[0] - Decimal128.One);
    }
    [Fact] public void InvalidRanksAndDimensionsAreRejected()
    {
        var vector = new double[] { 1, 2 }.ToTypedNDArray();
        Assert.Throws<ArgumentException>(() => vector.FactorizeLu());
        var identity = new double[,] { { 1, 0 }, { 0, 1 } }.ToTypedNDArray();
        Assert.Throws<ArgumentException>(() => identity.FactorizeLu().Solve(new double[,] { { 1, 2 } }.ToTypedNDArray()));
        Assert.Throws<ArgumentException>(() => identity.LeastSquares(new double[] { 1 }.ToTypedNDArray()));
    }
}
