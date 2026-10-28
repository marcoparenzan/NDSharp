using System.Numerics;
using NDSharp.Random;

namespace NDSharp.Tests;

public class EigTests
{
    static NDArray Real(double[,] m) => NDArray.FromMultiDim(m);

    static Complex[] Values(NDArray a)
    {
        if (a.DType.IsComplex()) return a.AsType(DType.Complex128).ToArray<Complex>();
        return a.AsType(DType.Float64).ToArray<double>().Select(x => new Complex(x, 0)).ToArray();
    }

    /// <summary>Greedy nearest-neighbour match of two eigenvalue lists (the order is unspecified); returns the worst distance.</summary>
    static double WorstMatch(Complex[] got, Complex[] want)
    {
        Assert.Equal(want.Length, got.Length);
        var free = got.ToList();
        double worst = 0;
        foreach (var e in want)
        {
            int best = 0;
            for (int i = 1; i < free.Count; i++) if (Complex.Abs(free[i] - e) < Complex.Abs(free[best] - e)) best = i;
            worst = Math.Max(worst, Complex.Abs(free[best] - e));
            free.RemoveAt(best);
        }
        return worst;
    }

    static double Norm(Complex[] v) => Math.Sqrt(v.Sum(z => z.Real * z.Real + z.Imaginary * z.Imaginary));

    /// <summary>max over the stack of || A v_i - w_i v_i || / (||A|| + 1) and the worst deviation of ||v_i|| from 1.</summary>
    static (double residual, double normError) Check(NDArray a, NDArray w, NDArray v)
    {
        int n = a.Shape[^1], count = a.Size / (n * n);
        var A = Values(a); var W = Values(w); var V = Values(v);
        double res = 0, nerr = 0;
        for (int t = 0; t < count; t++)
        {
            double an = Norm(A.Skip(t * n * n).Take(n * n).ToArray());
            for (int k = 0; k < n; k++)
            {
                var col = new Complex[n];
                for (int i = 0; i < n; i++) col[i] = V[t * n * n + i * n + k];
                nerr = Math.Max(nerr, Math.Abs(Norm(col) - 1));
                var r = new Complex[n];
                for (int i = 0; i < n; i++)
                {
                    Complex s = 0;
                    for (int j = 0; j < n; j++) s += A[t * n * n + i * n + j] * col[j];
                    r[i] = s - W[t * n + k] * col[i];
                }
                res = Math.Max(res, Norm(r) / (an + 1));
            }
        }
        return (res, nerr);
    }

    [Fact]
    public void RotationHasPurelyImaginaryEigenvalues()
    {
        var (w, v) = np.Eig(Real(new double[,] { { 0, -1 }, { 1, 0 } }));
        Assert.Equal(DType.Complex128, w.DType);
        Assert.True(WorstMatch(Values(w), new[] { new Complex(0, 1), new Complex(0, -1) }) < 1e-14);
        var (res, nerr) = Check(Real(new double[,] { { 0, -1 }, { 1, 0 } }), w, v);
        Assert.True(res < 1e-14 && nerr < 1e-14);
    }

    [Fact]
    public void RealSpectrumIsStillComplexTypedButExactlyReal()
    {
        // numpy >= 2.5 always returns complex; for a real spectrum the imaginary parts must be exactly zero
        var a = Real(new double[,] { { 2, 1 }, { 1, 3 } });
        var (w, v) = np.Eig(a);
        Assert.Equal(DType.Complex128, w.DType);
        Assert.Equal(DType.Complex128, v.DType);
        Assert.All(Values(w), z => Assert.Equal(0.0, z.Imaginary));
        Assert.All(Values(v), z => Assert.Equal(0.0, z.Imaginary));
        double r5 = Math.Sqrt(5);
        Assert.True(WorstMatch(Values(w), new Complex[] { (5 - r5) / 2, (5 + r5) / 2 }) < 1e-14);
    }

    [Fact]
    public void TriangularMatrixEigenvaluesAreTheDiagonal()
    {
        var a = Real(new double[,] { { 4, 7, -2 }, { 0, -3, 5 }, { 0, 0, 1.5 } });
        var (w, v) = np.Eig(a);
        Assert.True(WorstMatch(Values(w), new Complex[] { 4, -3, 1.5 }) < 1e-14);
        var (res, _) = Check(a, w, v);
        Assert.True(res < 1e-14);
    }

    [Fact]
    public void AgreesWithEighOnSymmetricMatrices()
    {
        var a = Real(new double[,] { { 4, 1, 2 }, { 1, 3, 0 }, { 2, 0, 5 } });
        var (wg, vg) = np.Eig(a);
        var (we, _) = np.Eigh(a);
        Assert.True(WorstMatch(Values(wg), Values(we)) < 1e-12);
        Assert.All(Values(wg), z => Assert.Equal(0.0, z.Imaginary));
        var (res, nerr) = Check(a, wg, vg);
        Assert.True(res < 1e-13 && nerr < 1e-13);
    }

    [Fact]
    public void EigvalsMatchesEig()
    {
        var rng = Generator.DefaultRng(new System.Numerics.BigInteger[] { 7 });
        var a = rng.StandardNormal(new[] { 7, 7 });
        Assert.True(WorstMatch(Values(np.Eigvals(a)), Values(np.Eig(a).w)) < 1e-12);
    }

    [Fact]
    public void StacksAreSolvedMatrixByMatrix()
    {
        var a = Generator.DefaultRng(new System.Numerics.BigInteger[] { 3 }).StandardNormal(new[] { 4, 5, 5 });
        var (w, v) = np.Eig(a);
        Assert.Equal(new[] { 4, 5 }, w.Shape);
        Assert.Equal(new[] { 4, 5, 5 }, v.Shape);
        var (res, nerr) = Check(a, w, v);
        Assert.True(res < 1e-12 && nerr < 1e-12);
    }

    [Fact]
    public void ComplexInputGivesComplexOutput()
    {
        var a = new NDArray(DType.Complex128, new[] { new Complex(1, 1), new Complex(2, 0), new Complex(0, -1), new Complex(3, 2) }, new[] { 2, 2 });
        var (w, v) = np.Eig(a);
        Assert.Equal(DType.Complex128, w.DType);
        var (res, nerr) = Check(a, w, v);
        Assert.True(res < 1e-13 && nerr < 1e-13);
        // trace and determinant: sum and product of the eigenvalues
        var W = Values(w);
        Assert.True(Complex.Abs(W[0] + W[1] - new Complex(4, 3)) < 1e-13);
        Assert.True(Complex.Abs(W[0] * W[1] - (new Complex(1, 1) * new Complex(3, 2) - new Complex(2, 0) * new Complex(0, -1))) < 1e-13);
    }

    [Fact]
    public void Float32InputGivesSinglePrecisionOutput()
    {
        var a = Real(new double[,] { { 1, 2 }, { 3, 4 } }).AsType(DType.Float32);
        Assert.Equal(DType.Complex64, np.Eig(a).w.DType);
        Assert.Equal(DType.Complex64, np.Eig(a).v.DType);
        var rot = Real(new double[,] { { 0, -1 }, { 1, 0 } }).AsType(DType.Float32);
        Assert.Equal(DType.Complex64, np.Eig(rot).w.DType);
    }

    [Fact]
    public void IntegerInputIsPromotedToDouble()
    {
        var a = NDArray.FromArray(new long[] { 2, 0, 0, 3 }, 2, 2);
        var w = np.Eigvals(a);
        Assert.Equal(DType.Complex128, w.DType);
        Assert.True(WorstMatch(Values(w), new Complex[] { 2, 3 }) < 1e-15);
    }

    [Fact]
    public void ShapesAndEdgeCases()
    {
        Assert.Equal(new[] { 0 }, np.Eigvals(np.Zeros(new[] { 0, 0 })).Shape);
        var (w, v) = np.Eig(Real(new double[,] { { 4.25 } }));
        Assert.Equal(new Complex(4.25, 0), Values(w)[0]);
        Assert.Equal(1.0, Complex.Abs(Values(v)[0]));
    }

    [Fact]
    public void RejectsNonSquareAndNonFinite()
    {
        Assert.Throws<NDLinAlgException>(() => np.Eig(np.Zeros(new[] { 2, 3 })));
        Assert.Throws<NDLinAlgException>(() => np.Eig(np.Zeros(new[] { 3 })));
        Assert.Throws<NDLinAlgException>(() => np.Eigvals(Real(new double[,] { { 1, double.NaN }, { 0, 1 } })));
        Assert.Throws<NDLinAlgException>(() => np.Eigvals(Real(new double[,] { { 1, double.PositiveInfinity }, { 0, 1 } })));
    }

    [Fact]
    public void LargeMatrixConvergesAndIsAccurate()
    {
        var a = Generator.DefaultRng(new System.Numerics.BigInteger[] { 99 }).StandardNormal(new[] { 120, 120 });
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var (w, v) = np.Eig(a);
        Assert.True(sw.Elapsed < TimeSpan.FromSeconds(20), $"took {sw.Elapsed}");
        var (res, nerr) = Check(a, w, v);
        Assert.True(res < 1e-10 && nerr < 1e-12, $"residual {res}, norm error {nerr}");
        // trace = sum of eigenvalues
        double trace = 0; var A = Values(a);
        for (int i = 0; i < 120; i++) trace += A[i * 120 + i].Real;
        Assert.True(Math.Abs(Values(w).Sum(z => z.Real) - trace) < 1e-9);
    }

    public static IEnumerable<object[]> OracleCases() => EigCases.All.Select(c => new object[] { c.Name });

    [Theory]
    [MemberData(nameof(OracleCases))]
    public void MatchesNumpy(string name)
    {
        var c = EigCases.All.Single(x => x.Name == name);
        NDArray a;
        if (c.Complex)
            a = new NDArray(DType.Complex128, c.Re.Zip(c.Im, (r, i) => new Complex(r, i)).ToArray(), c.Shape);
        else
            a = new NDArray(DType.Float64, c.Re, c.Shape);

        var (w, v) = np.Eig(a);
        int n = c.Shape[^1], count = a.Size / (n * n);
        var got = Values(w);
        var want = c.ExpRe.Zip(c.ExpIm, (r, i) => new Complex(r, i)).ToArray();

        double scale = 1 + Norm(Values(a));
        for (int t = 0; t < count; t++)
        {
            double worst = WorstMatch(got.Skip(t * n).Take(n).ToArray(), want.Skip(t * n).Take(n).ToArray());
            Assert.True(worst <= c.Tol * scale, $"{name}[{t}]: eigenvalue deviates from numpy by {worst:E2} (tolerance {c.Tol * scale:E2})");
        }

        // numpy >= 2.5 always returns complex; a real spectrum of a real matrix must still have exactly zero imaginary parts
        Assert.Equal(DType.Complex128, w.DType);
        if (!c.Complex && c.ExpIm.All(x => x == 0)) Assert.All(got, z => Assert.True(Math.Abs(z.Imaginary) <= c.Tol * scale, $"{name}: spurious imaginary part {z.Imaginary:E2}"));

        // eigenvectors are only defined up to a phase: check them through the residual
        var (res, nerr) = Check(a, w, v);
        double resTol = name.Contains("defective") || name.StartsWith("nilpotent") ? 1e-6 : 1e-9;
        Assert.True(res < resTol, $"{name}: residual {res:E2}");
        Assert.True(nerr < 1e-12, $"{name}: vector norm error {nerr:E2}");
    }
}




