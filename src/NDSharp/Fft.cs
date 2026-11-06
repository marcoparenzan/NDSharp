// Copyright (c) 2026 Marco Parenzan
//
// Licensed under the MIT License. See the LICENSE file in the project
// root for full license information.

using System.Numerics;
using MathemaSharp.Kernels;

namespace NDSharp;

/// <summary><c>numpy.fft</c>. The transform of one line (radix-2 for power-of-two lengths, Bluestein otherwise, exact twiddle factors, the Hermitian completion of real data, the
/// half-spectrum inverse, the scales and the frequencies) is <c>MathemaSharp.Kernels.Fourier</c>; this class does what the arrays need: axes, padding and cropping, dtypes,
/// the multi-dimensional passes.</summary>
public static class Fft
{
    private static DType OutputType(NDArray a) => a.DType is DType.Float32 or DType.Float16 or DType.Complex64
        or DType.Int8 or DType.UInt8 or DType.Int16 or DType.UInt16 or DType.Bool ? DType.Complex64 : DType.Complex128;

    private static double Scale(string? norm, int n, bool inverse) => Fourier.Scale(norm switch
    {
        null or "backward" => FourierNorm.Backward,
        "ortho" => FourierNorm.Ortho,
        "forward" => FourierNorm.Forward,
        _ => throw new NDValueException($"Invalid norm value {norm}; should be \"backward\", \"ortho\" or \"forward\"."),
    }, n, inverse);

    /// <summary>Crops or zero-pads <paramref name="a"/> to <paramref name="n"/> along <paramref name="axis"/>.</summary>
    private static NDArray Resize(NDArray a, int n, int axis)
    {
        int cur = a.Shape[axis];
        if (cur == n) return a;
        var idx = Enumerable.Repeat<NDIndex>(Slice.All, a.Ndim).ToArray();
        if (n < cur)
        {
            idx[axis] = new Slice(0, n);
            return a.Get(idx);
        }
        var shape = (int[])a.Shape.Clone();
        shape[axis] = n;
        var z = np.Zeros(shape, a.DType);
        idx[axis] = new Slice(0, cur);
        z.Put(a, idx);
        return z;
    }

    private static NDArray Along(NDArray input, int? n, int axis, string? norm, bool inverse)
    {
        if (input.Ndim == 0) throw new NDValueException("Invalid number of FFT data points (0) specified.");
        int ax = np.NormalizeAxis(axis, input.Ndim);
        var outType = OutputType(input);
        var a = input.CastTo(outType);
        int len = n ?? a.Shape[ax];
        if (len < 1) throw new NDValueException($"Invalid number of FFT data points ({len}) specified.");
        a = Resize(a, len, ax);
        var moved = np.MoveAxis(a, ax, -1).Copy();
        var buf = (Complex[])moved.Buffer;
        double scale = Scale(norm, len, inverse);
        bool realInput = !input.DType.IsComplex() && !inverse;
        var line = new Complex[len];
        for (int s = 0; s < buf.Length; s += len)
        {
            Array.Copy(buf, s, line, 0, len);
            Fourier.Transform(line, inverse);
            if (realInput) Fourier.MakeHermitian(line);
            for (int i = 0; i < len; i++) buf[s + i] = scale == 1.0 ? line[i] : line[i] * scale;
        }
        if (outType == DType.Complex64) Cx.RoundC64(buf);
        return np.MoveAxis(moved, -1, ax).Copy();
    }

    public static NDArray FFT(NDArray a, int? n = null, int axis = -1, string? norm = null) => Along(a, n, axis, norm, false);
    public static NDArray IFFT(NDArray a, int? n = null, int axis = -1, string? norm = null) => Along(a, n, axis, norm, true);

    private static NDArray Multi(NDArray a, int[]? s, int[]? axes, string? norm, bool inverse, int defaultAxes)
    {
        int[] ax = axes ?? (defaultAxes > 0
            ? Enumerable.Range(a.Ndim - defaultAxes, defaultAxes).ToArray()
            : Enumerable.Range(0, a.Ndim).ToArray());
        if (s is not null && s.Length != ax.Length)
            throw new NDValueException("Shape and axes have different lengths.");
        var r = a;
        for (int i = ax.Length - 1; i >= 0; i--)
            r = Along(r, s?[i], ax[i], norm, inverse);
        return r;
    }

    public static NDArray FFT2(NDArray a, int[]? s = null, int[]? axes = null, string? norm = null) => Multi(a, s, axes ?? new[] { -2, -1 }, norm, false, 0);
    public static NDArray IFFT2(NDArray a, int[]? s = null, int[]? axes = null, string? norm = null) => Multi(a, s, axes ?? new[] { -2, -1 }, norm, true, 0);
    public static NDArray FFTN(NDArray a, int[]? s = null, int[]? axes = null, string? norm = null) => Multi(a, s, axes, norm, false, 0);
    public static NDArray IFFTN(NDArray a, int[]? s = null, int[]? axes = null, string? norm = null) => Multi(a, s, axes, norm, true, 0);

    /// <summary>numpy <c>rfft</c>: the first n/2+1 coefficients of the transform of real input.</summary>
    public static NDArray RFFT(NDArray a, int? n = null, int axis = -1, string? norm = null)
    {
        if (a.DType.IsComplex()) throw new NDTypeException("rfft requires real input");
        var full = Along(a, n, axis, norm, false); // real input: already exactly Hermitian (see MakeHermitian)
        int ax = np.NormalizeAxis(axis, a.Ndim);
        int keep = full.Shape[ax] / 2 + 1;
        var idx = Enumerable.Repeat<NDIndex>(Slice.All, full.Ndim).ToArray();
        idx[ax] = new Slice(0, keep);
        return full.Get(idx).Copy();
    }

    /// <summary>numpy <c>irfft</c>: real signal from a Hermitian half-spectrum.</summary>
    public static NDArray IRFFT(NDArray a, int? n = null, int axis = -1, string? norm = null)
    {
        int ax = np.NormalizeAxis(axis, a.Ndim);
        int m = a.Shape[ax];
        int len = n ?? 2 * (m - 1);
        if (len < 1) throw new NDValueException($"Invalid number of FFT data points ({len}) specified.");
        var outType = OutputType(a);
        var half = Resize(a.CastTo(outType), len / 2 + 1, ax);
        var moved = np.MoveAxis(half, ax, -1).Copy();
        var src = (Complex[])moved.Buffer;
        int hl = len / 2 + 1;
        int lines = src.Length / hl;
        var result = new double[lines * len];
        double scale = Scale(norm, len, true);
        for (int l = 0; l < lines; l++) Fourier.InverseReal(src.AsSpan(l * hl, hl), result.AsSpan(l * len, len), scale);
        var shape = moved.Shape[..^1].Concat(new[] { len }).ToArray();
        var r = new NDArray(DType.Float64, result, shape);
        var real = outType == DType.Complex64 ? r.AsType(DType.Float32) : r;
        return np.MoveAxis(real, -1, ax).Copy();
    }

    public static NDArray FFTShift(NDArray a, int[]? axes = null)
    {
        var ax = axes ?? Enumerable.Range(0, a.Ndim).ToArray();
        var r = a;
        foreach (var x in ax) r = np.Roll(r, a.Shape[np.NormalizeAxis(x, a.Ndim)] / 2, x);
        return r;
    }

    public static NDArray IFFTShift(NDArray a, int[]? axes = null)
    {
        var ax = axes ?? Enumerable.Range(0, a.Ndim).ToArray();
        var r = a;
        foreach (var x in ax) r = np.Roll(r, -(a.Shape[np.NormalizeAxis(x, a.Ndim)] / 2), x);
        return r;
    }

    /// <summary>numpy <c>fftfreq(n, d)</c>.</summary>
    public static NDArray FFTFreq(int n, double d = 1.0)
    {
        if (n <= 0) throw new NDValueException("n should be an integer greater than 0");
        var r = new double[n];
        Fourier.Frequencies(r, d);
        return NDArray.FromArray(r, n);
    }

    public static NDArray RFFTFreq(int n, double d = 1.0)
    {
        if (n <= 0) throw new NDValueException("n should be an integer greater than 0");
        var r = new double[n / 2 + 1];
        Fourier.RealFrequencies(r, n, d);
        return NDArray.FromArray(r, r.Length);
    }
}
