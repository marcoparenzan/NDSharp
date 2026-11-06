// Copyright (c) 2026 Marco Parenzan
// Licensed under the MIT License. See the LICENSE file in the project root.
using System.Numerics;
using NDSharp;
using Xunit;

namespace NDSharp.Tests;

/// <summary>numpy.fft on NDSharp arrays (the transform itself is MathemaSharp.Kernels.Fourier). The results are fingerprints of what the implementation that lived in
/// NDSharp before the move produced (see <see cref="Golden"/>); the analytic tests below check the values independently.</summary>
public class FftTests
{
    static NDArray Real(DType dtype, params int[] shape)
    {
        int n = 1; foreach (var s in shape) n *= s;
        var d = new double[n];
        for (int i = 0; i < n; i++) d[i] = Math.Round(Math.Sin(i * 0.77) * 5 + (i % 4), 3);
        return NDArray.FromArray(d, shape).AsType(dtype);
    }

    static NDArray Cplx(DType dtype, params int[] shape)
    {
        int n = 1; foreach (var s in shape) n *= s;
        var d = new Complex[n];
        for (int i = 0; i < n; i++) d[i] = new Complex(Math.Round(Math.Sin(i * 0.77) * 5, 3), Math.Round(Math.Cos(i * 1.1) * 3, 3));
        return NDArray.FromArray(d, shape).AsType(dtype);
    }

    static void Fingerprint(string key, Func<NDArray> call)
    {
        string actual;
        try { actual = Golden.Hash(call()); } catch (Exception e) { actual = Golden.Describe(e); }
        Golden.Check("Fft|" + key, actual, key);
    }

    public static IEnumerable<object[]> Types() => new[] { DType.Float64, DType.Float32, DType.Int32, DType.Bool, DType.Complex128, DType.Complex64 }.Select(t => new object[] { t });

    static NDArray Make(DType t, params int[] shape) => t.IsComplex() ? Cplx(t, shape) : Real(t, shape);

    [Theory, MemberData(nameof(Types))]
    public void OneDimensionalTransformsOfEveryLength(DType dtype)
    {
        foreach (int n in new[] { 1, 2, 3, 4, 5, 7, 8, 12, 16, 17, 31, 64, 100, 128, 255 })
        {
            var a = Make(dtype, n);
            Fingerprint($"fft {dtype} n={n}", () => Fft.FFT(a));
            Fingerprint($"ifft {dtype} n={n}", () => Fft.IFFT(a));
            if (!dtype.IsComplex())
            {
                Fingerprint($"rfft {dtype} n={n}", () => Fft.RFFT(a));
                Fingerprint($"irfft {dtype} n={n}", () => Fft.IRFFT(a));
            }
        }
    }

    [Theory, MemberData(nameof(Types))]
    public void AxesPaddingCroppingAndNorms(DType dtype)
    {
        var a = Make(dtype, 3, 5, 6);
        foreach (var axis in new[] { 0, 1, 2, -1 })
        foreach (var norm in new string?[] { null, "backward", "ortho", "forward" })
        {
            Fingerprint($"fft {dtype} axis={axis} norm={norm}", () => Fft.FFT(a, null, axis, norm));
            Fingerprint($"ifft {dtype} axis={axis} norm={norm}", () => Fft.IFFT(a, null, axis, norm));
        }
        foreach (var n in new[] { 1, 4, 5, 9, 16 })
        {
            Fingerprint($"fft {dtype} n={n} axis=1", () => Fft.FFT(a, n, 1));
            if (!dtype.IsComplex()) Fingerprint($"rfft {dtype} n={n} axis=0", () => Fft.RFFT(a, n, 0));
        }
        Fingerprint($"fft2 {dtype}", () => Fft.FFT2(a));
        Fingerprint($"ifft2 {dtype}", () => Fft.IFFT2(a));
        Fingerprint($"fftn {dtype}", () => Fft.FFTN(a));
        Fingerprint($"ifftn {dtype} axes 0,2", () => Fft.IFFTN(a, null, new[] { 0, 2 }));
        Fingerprint($"fftn {dtype} s", () => Fft.FFTN(a, new[] { 4, 4, 8 }));
        Fingerprint($"fftshift {dtype}", () => Fft.FFTShift(a));
        Fingerprint($"ifftshift {dtype} axes 1", () => Fft.IFFTShift(a, new[] { 1 }));
    }

    [Fact]
    public void RealInputRoundTripsThroughTheHalfSpectrum()
    {
        foreach (int n in new[] { 2, 5, 8, 9, 31, 64 })
        {
            var a = Real(DType.Float64, n);
            var back = Fft.IRFFT(Fft.RFFT(a), n);
            var x = a.Copy().Data<double>(); var y = back.Copy().Data<double>();
            for (int i = 0; i < n; i++) Assert.Equal(x[i], y[i], 11);
        }
    }

    [Fact]
    public void KnownTransforms()
    {
        var signal = NDArray.FromArray(new double[] { 0, 1, 0, -1 }, 4);
        var spectrum = Fft.FFT(signal).Copy().Data<Complex>();
        Assert.Equal(new Complex(0, 0), spectrum[0]);
        Assert.Equal(new Complex(0, -2), spectrum[1]);
        Assert.Equal(new Complex(0, 0), spectrum[2]);
        Assert.Equal(new Complex(0, 2), spectrum[3]);
        Assert.Equal(new[] { 0.0, 0.25, -0.5, -0.25 }, Fft.FFTFreq(4).Copy().Data<double>());
        Assert.Equal(new[] { 0.0, 0.25, 0.5 }, Fft.RFFTFreq(4).Copy().Data<double>());
        Assert.Equal(DType.Complex64, Fft.FFT(Real(DType.Float32, 8)).DType);
        Assert.Equal(DType.Complex128, Fft.FFT(Real(DType.Int32, 8)).DType == DType.Complex128 ? DType.Complex128 : DType.Complex64);
    }

    [Fact]
    public void Errors()
    {
        Assert.Throws<NDValueException>(() => Fft.FFT(Real(DType.Float64, 4), 0));
        Assert.Throws<NDValueException>(() => Fft.FFT(Real(DType.Float64, 4), null, -1, "sideways"));
        Assert.Throws<NDValueException>(() => Fft.FFT(NDArray.Scalar(1.0)));
        Assert.Throws<NDTypeException>(() => Fft.RFFT(Cplx(DType.Complex128, 4)));
        Assert.Throws<NDValueException>(() => Fft.FFTFreq(0));
        Assert.Throws<NDValueException>(() => Fft.FFT2(Real(DType.Float64, 3, 3), new[] { 2 }));
    }
}
