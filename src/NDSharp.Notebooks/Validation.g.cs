#nullable enable
using System;
using System.Linq;
using System.IO;
using NDSharp;
using NDSharp.Frame;
using NDSharp.Plot;
using NDSharp.Image;
using NDSharp.Wavelets;
using SkiaSharp;
using FIndex = NDSharp.Frame.Index;
public static class Validation { public static void Run() {
Sample0(); Console.WriteLine("PASS 01-arrays");
Sample1(); Console.WriteLine("PASS 02-views");
Sample2(); Console.WriteLine("PASS 03-broadcasting");
Sample3(); Console.WriteLine("PASS 04-reductions");
Sample4(); Console.WriteLine("PASS 05-linear-algebra");
Sample5(); Console.WriteLine("PASS 06-random");
Sample6(); Console.WriteLine("PASS 07-signal");
Sample7(); Console.WriteLine("PASS 08-series");
Sample8(); Console.WriteLine("PASS 09-csv");
Sample9(); Console.WriteLine("PASS 10-joins");
Sample10(); Console.WriteLine("PASS 11-line-plot");
Sample11(); Console.WriteLine("PASS 12-heatmap");
Sample12(); Console.WriteLine("PASS 13-image");
Sample13(); Console.WriteLine("PASS 14-wavelets");
}
public static void Sample0() {
var a = np.Reshape(np.Arange(0L, 6L), 2, 3);
Console.WriteLine(a);
Console.WriteLine($"shape={string.Join(",", a.Shape)}; dtype={a.DType}; size={a.Size}");
Console.WriteLine(a.AsType(DType.Float32));
if (a.Size != 6) throw new Exception("Unexpected size");
}
public static void Sample1() {
var a = np.Reshape(np.Arange(0L, 12L), 3, 4);
var view = a.Get(new Slice(0, 2), new Slice(1, 3));
Console.WriteLine(view);
Console.WriteLine($"Shared buffer: {ReferenceEquals(a.Buffer, view.Buffer)}");
Console.WriteLine(np.Transpose(a));
var independent = view.Copy();
if (ReferenceEquals(independent.Buffer, a.Buffer)) throw new Exception("Copy must own storage");
}
public static void Sample2() {
var a = np.Reshape(np.Arange(0L, 6L), 2, 3);
var offset = np.Arange(10L, 13L);
var result = np.Add(a, offset);
Console.WriteLine(result);
Console.WriteLine(np.Multiply(result, NDArray.Scalar(2L)));
if (Convert.ToInt64(result[1, 2]) != 17L) throw new Exception("Broadcast mismatch");
}
public static void Sample3() {
var a = np.Reshape(np.Arange(1L, 7L), 2, 3);
Console.WriteLine(np.Sum(a));
Console.WriteLine(np.Mean(a));
Console.WriteLine(np.Sum(a, axis: new[] { 1 }, keepdims: true));
Console.WriteLine(np.Median(a));
if (Convert.ToInt64(np.Sum(a).GetAt(0)) != 21L) throw new Exception("Sum mismatch");
}
public static void Sample4() {
var a = new NDArray(DType.Float64, new double[] { 2, 0, 0, 4 }, new[] { 2, 2 });
var b = new NDArray(DType.Float64, new double[] { 4, 12 }, new[] { 2 });
var x = np.Solve(a, b);
Console.WriteLine(x);
Console.WriteLine(np.Det(a));
if (Math.Abs(Convert.ToDouble(x[1]) - 3) > 1e-10) throw new Exception("Solve mismatch");
}
public static void Sample5() {
var seed = new System.Numerics.BigInteger[] { 42 };
var rng = NDSharp.Random.Generator.DefaultRng(seed);
var a = rng.Random(new[] { 2, 3 });
Console.WriteLine(a);
Console.WriteLine(rng.StandardNormal(new[] { 5 }));
var repeat = NDSharp.Random.Generator.DefaultRng(seed).Random(new[] { 2, 3 });
if (!(bool)np.All(np.Equal(a, repeat)).GetAt(0)) throw new Exception("Seed mismatch");
}
public static void Sample6() {
var signal = new NDArray(DType.Float64, new double[] { 0, 1, 0, -1 }, new[] { 4 });
var spectrum = Fft.FFT(signal);
Console.WriteLine(spectrum);
Console.WriteLine(Fft.IFFT(spectrum));
Console.WriteLine(Fft.FFTFreq(4));
Console.WriteLine(np.Diff(signal));
}
public static void Sample7() {
var a = new Series(Column.FromLongs(new long[] { 1, 2 }), FIndex.OfStrings(new[] { "x", "y" }));
var b = new Series(Column.FromLongs(new long[] { 10, 20 }), FIndex.OfStrings(new[] { "y", "z" }));
Console.WriteLine(Formatter.SeriesRepr(Ops.Binary(BinOp.Add, a, b), new DisplayOptions()));
var values = Column.FromDoubles(new[] { 1.0, double.NaN, 3.0 });
Console.WriteLine(NDSharp.Frame.Reduce.Scalar("mean", values));
if (!Equals(NDSharp.Frame.Reduce.Scalar("count", values), 2L)) throw new Exception("Missing count mismatch");
}
public static void Sample8() {
var csv = "city,value\nParis,10\nRome,20\n";
var (columns, names) = Csv.Read(csv, new Csv.ReadOptions());
var frame = new DataFrame(columns, FIndex.Of(names));
Console.WriteLine(Formatter.FrameRepr(frame));
Console.WriteLine(frame.GetColumn((object?)"value"));
if (frame.NRows != 2) throw new Exception("Row count mismatch");
}
public static void Sample9() {
var left = new DataFrame(new[] { Column.FromLongs(new long[] { 1, 2 }), Column.FromLongs(new long[] { 10, 20 }) }, FIndex.OfStrings(new[] { "key", "a" }));
var right = new DataFrame(new[] { Column.FromLongs(new long[] { 2, 3 }), Column.FromLongs(new long[] { 5, 6 }) }, FIndex.OfStrings(new[] { "key", "b" }));
var joined = Merge.Join(left, right, new Merge.Spec { LeftOn = new object?[] { "key" }, RightOn = new object?[] { "key" }, How = "outer" });
Console.WriteLine(Formatter.FrameRepr(joined));
if (joined.NRows != 3) throw new Exception("Join mismatch");
}
public static void Sample10() {
var figure = new Figure();
var axes = figure.AddAxes(GridCell.Single(1, 1, 0));
axes.Add(new Line2D(new double[] { 0, 1, 2, 3 }, new double[] { 0, 1, 4, 9 }, SKColors.Blue));
var png = figure.RenderPng();
File.WriteAllBytes("ndsharp-line.png", png);
Console.WriteLine(Path.GetFullPath("ndsharp-line.png"));
if (png[0] != 0x89) throw new Exception("PNG mismatch");
}
public static void Sample11() {
var figure = new Figure();
var axes = figure.AddAxes(GridCell.Single(1, 1, 0));
var image = new ImageArtist { Rows = 2, Cols = 3, Values = new double[] { 0, 1, 2, 3, 4, 5 }, Cmap = Colormap.Get("viridis") };
axes.Add(image);
figure.AddColorbar(image, label: "value");
File.WriteAllBytes("ndsharp-heatmap.png", figure.RenderPng());
Console.WriteLine(Path.GetFullPath("ndsharp-heatmap.png"));
}
public static void Sample12() {
var bgr = np.Full(new[] { 2, 2, 3 }, 100L, DType.UInt8);
var gray = Cv.CvtColor(bgr, 6); // OpenCV COLOR_BGR2GRAY
Console.WriteLine(gray);
using var mat = Mats.ToMat(gray);
var restored = Mats.FromMat(mat);
if (!(bool)np.All(np.Equal(gray, restored)).GetAt(0)) throw new Exception("Mat roundtrip mismatch");
}
public static void Sample13() {
var signal = new double[] { 1, 2, 3, 4, 5, 6, 7, 8 };
var wavelet = Wavelet.Get("haar");
var (approximation, detail) = Dwt.Dwt1D(signal, wavelet);
Console.WriteLine(string.Join(", ", approximation));
Console.WriteLine(string.Join(", ", detail));
var restored = Dwt.Idwt1D(approximation, detail, wavelet);
if (signal.Zip(restored).Any(p => Math.Abs(p.First - p.Second) > 1e-10)) throw new Exception("Wavelet roundtrip mismatch");
}
}
