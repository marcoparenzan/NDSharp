// Copyright (c) 2026 Marco Parenzan
// Licensed under the MIT License. See the LICENSE file in the project root.
using System.Numerics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using NDSharp;

namespace NDSharp.Tests;

/// <summary>Reference results of the kernel-path tests, kept as short fingerprints (Fixtures/kernelpath-golden.txt). Each case has a key (test name and label) and a fingerprint
/// of the result: dtype, shape and a hash of the bytes (every NaN counts as the same NaN; the two zeros differ).
/// <para>The references were produced on 2026-11-06 by NDSharp's former general engine (git tag <c>engine-before-removal-20261106</c>), and in the same run every kernel
/// result was required to agree with the engine element by element. A failure here means a kernel result changed since, or the case is new.</para>
/// <para>To add new cases, check their results by hand (against numpy or Octave, or a formula), then run the tests with <c>NDSHARP_GOLDEN=add</c> and
/// <c>NDSHARP_GOLDEN_FILE</c> set to the path of <c>Fixtures/kernelpath-golden.txt</c> in the source tree: unknown keys are appended with the current result.</para></summary>
internal static class Golden
{
    static readonly object gate = new();
    static Dictionary<string, string>? table;

    static bool Adding => Environment.GetEnvironmentVariable("NDSHARP_GOLDEN") == "add";

    static Dictionary<string, string> Table()
    {
        if (table is not null) return table;
        lock (gate)
        {
            if (table is not null) return table;
            var t = new Dictionary<string, string>();
            var path = Path.Combine(AppContext.BaseDirectory, "Fixtures", "kernelpath-golden.txt");
            foreach (var line in File.ReadLines(path))
            {
                int tab = line.IndexOf('\t');
                if (tab > 0) t[line[..tab]] = line[(tab + 1)..];
            }
            table = t;
            return t;
        }
    }

    /// <summary>Fails the test unless <paramref name="actual"/> is the reference fingerprint of <paramref name="key"/>.</summary>
    public static void Check(string key, string actual, string what)
    {
        string? reference;
        lock (gate) Table().TryGetValue(key, out reference);
        if (reference is null)
        {
            if (!Adding) throw new Xunit.Sdk.XunitException($"no reference result for '{key}' in Fixtures/kernelpath-golden.txt (check the case by hand, then add it with NDSHARP_GOLDEN=add)");
            lock (gate)
            {
                var file = Environment.GetEnvironmentVariable("NDSHARP_GOLDEN_FILE") ?? throw new InvalidOperationException("NDSHARP_GOLDEN_FILE is not set");
                File.AppendAllText(file, key + "\t" + actual + "\n");
                Table()[key] = actual;
            }
            return;
        }
        if (reference != actual) throw new Xunit.Sdk.XunitException($"{what}: the result differs from the reference ({reference} vs {actual})");
    }

    public static string Describe(Exception e) => "ERR|" + e.GetType().Name + "|" + e.Message.Replace('\n', ' ').Replace('\t', ' ');

    public static string Hash(NDArray array)
    {
        var a = array.Copy();
        var bytes = Bytes(a.Buffer);
        var h = SHA256.HashData(bytes);
        return $"{a.DType}|{string.Join(",", a.Shape)}|{Convert.ToHexString(h, 0, 8)}";
    }

    public static string Hash(IReadOnlyList<NDArray> arrays) => string.Join(";", arrays.Select(Hash));

    static byte[] Raw<T>(T[] a) where T : unmanaged => MemoryMarshal.AsBytes(a.AsSpan()).ToArray();

    static byte[] Bytes(Array buffer) => buffer switch
    {
        double[] d => Raw(d.Select(v => double.IsNaN(v) ? BitConverter.Int64BitsToDouble(0x7FF8000000000000) : v).ToArray()),
        float[] f => Raw(f.Select(v => float.IsNaN(v) ? BitConverter.Int32BitsToSingle(0x7FC00000) : v).ToArray()),
        Half[] h => Raw(h.Select(v => Half.IsNaN(v) ? Half.NaN : v).ToArray()),
        Complex[] z => Raw(z.Select(v => new Complex(double.IsNaN(v.Real) ? double.NaN : v.Real, double.IsNaN(v.Imaginary) ? double.NaN : v.Imaginary)).ToArray()),
        sbyte[] x => Raw(x),
        byte[] x => Raw(x),
        short[] x => Raw(x),
        ushort[] x => Raw(x),
        int[] x => Raw(x),
        uint[] x => Raw(x),
        long[] x => Raw(x),
        ulong[] x => Raw(x),
        bool[] x => Raw(x),
        _ => throw new NotSupportedException(buffer.GetType().Name),
    };
}
