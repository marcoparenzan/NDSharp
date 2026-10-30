# Usage

## Installation

NDSharp is a set of **libraries** plus the global tool **`ndsharp`**. Requires the **.NET 11 SDK**.

```powershell
# from the local feed (nothing is on nuget.org yet)
dotnet add package NDSharp        --source D:\Dev\NuGetLocalFeed --prerelease
dotnet add package NDSharp.Frame  --source D:\Dev\NuGetLocalFeed --prerelease   # DataFrame
dotnet add package NDSharp.Plot   --source D:\Dev\NuGetLocalFeed --prerelease   # plotting (SkiaSharp natives)
# also: NDSharp.Image (OpenCV), NDSharp.Wavelets
```

### The `ndsharp` tool

```powershell
dotnet tool install --global NDSharp.Cli --add-source D:\Dev\NuGetLocalFeed --prerelease
ndsharp info data.csv               # columns, types, non-null counts
ndsharp head data.csv -n 10         # first rows
ndsharp describe data.csv           # count, mean, std, quartiles per column
ndsharp convert data.csv out.json   # or out.csv
ndsharp selftest
```

Reads CSV (`--sep` for another separator); writes JSON records or CSV. Exit codes: 0 ok, 1 unknown command, 2 usage or input
error. Build and test from source: `dotnet test src/NDSharp.Cli.Tests`.

The whole suite, including these packages, can be downloaded with `D:\Dev\Science\ScienceSharp\Install-Suite.ps1 -Download`.
Notebooks use the JupyterSharp VS Code extension (see the suite [USAGE](../ScienceSharp/USAGE.md)). The sections below describe
the build from source, the libraries and the validation history.

## Current complete Science baseline

After physical filled sampling and band lookup delegation, Science gate --no-images passed31/31: D:\Dev\BuildArtifacts\ecosystem-validation\20261028-215324. Includes targeted/shared tests, package/source kernels, real notebook line/fill outputs and the rest of ecosystem checks. No new review image/screenshot files; notebook PNG outputs remain embedded.
This supersedes earlier no-full-gate notes, not full-family migration/release or all-input accessibility/pixel equivalence. Build warnings and suspended tests remain reported.

## Filled-contour band lookup now delegated

ContourBands selects inclusive bands and computes original midpoint normalization; internal exact boundaries/repeated levels choose the last matching band, final bound included, outside/non-finite data gives no band. Requires finite nondecreasing level snapshot; duplicates and a singleton retain legacy behavior. Palette choice, alpha and Skia rendering stay NDSharp.
Targeted builds/tests227/227 shared and19/19 ND passed; rebuilt source kernel notebook emits the expected line/band outputs, no review files. Not new full gate or release, complete family migration still open.

## Filled-contour physical sampler now shared

DrawFilled uses a RectilinearFieldSampler snapshot for binary-search bilinear values; band lookup, midpoint color selection, alpha, bitmap and Skia drawing stay native. 81 pre-migration private-Sample queries match exactly. Axes require ≥2 finite strictly increasing values and matching row-major data; non-finite sample values propagate, finite outside queries extrapolate as before. Descending line-contour geometry remains a separate supported path.
Targeted tests211/211 shared and19/19 ND passed. Rebuilt --source-plots kernel and real notebook line/bands outputs verified, no saved review images. Not new full gate, full contourf equivalence or release.

## Real-kernel contour integration — explicit source profile

From MathemaSharp run BuildPySharpKernel.cs with --source-plots, then RunNDSharpContourNotebook.cs. The Matplotlib binding defaults to the previous NuGet preview; ScienceUseSourcePlots=true explicitly references active NDSharp.Plot. A separate kernel-source-plots output prevents replacing the default package kernel.
Notebook NDSharp/pysharp/02-contours-kernel.ipynb executes in real JupyterNet/PySharp (.NET11), not native JupyterSharp. DLL metadata verifies the PlotSharp dependency, then one line and one filled contour produce embedded PNG320×240. No review PNG/screenshot files. Latest full gate --no-images: D:\Dev\BuildArtifacts\ecosystem-validation\20261028-203820, 31/31 exit0. Family migration/publication and CPython pixel equivalence remain separate.

## Line contour geometry now shared — source-build only

NDSharp.Plot references PlotSharp for RectilinearContours.Trace; Figure/Axes, Skia strokes/projection, color selection, automatic levels and filled contours stay native. Shared ContourKernel preserves ND's forward top/left edge arithmetic, fixed pairing, exact-vertex segments and non-finite corner-sum skipping. X/Y remain physical/nonuniform, not globally scaled indices.
Targeted build and 18/18 NDSharp.Plot tests passed; 204/204 shared PlotSharp tests passed including exact legacy geometry comparisons. No review images saved, full solution/notebook/package validation not claimed. Old preview packages are not updated by this source change.

## NDSharp.Plot pre-migration contour baseline

The active Skia renderer is unchanged and has no new PlotSharp dependency. Four baseline cases in ContourBaselineTests cover nonuniform physical X/Y, invalid-cell skipping, endpoint-level suppression and the separate filled-band renderer. Build/test passed: 18/18 tests. PNGs are rendered/inspected in memory, not saved.
From MathemaSharp, `dotnet run --file ops/InventoryNDSharpPlot.cs` records source/dependency hashes and repeats the targeted test run (requires tests built first). Report: ops/inventory/ndsharp-plot-contract.json. Candidate migration is line-contour geometry only; this is not a migration or full-solution/visual validation.

## Requirements and installation

Use the .NET 11 SDK for development and a compatible .NET runtime for library consumers. Commands below run from the repository root unless stated otherwise.

### Reference source projects

In your own .NET 11 project, reference only the libraries you need (adjust paths to your checkout):

```powershell
dotnet add reference ../NDSharp/src/NDSharp/NDSharp.csproj
dotnet add reference ../NDSharp/src/NDSharp.Plot/NDSharp.Plot.csproj
```

Companion projects reference the core automatically.

### Install packages

If the packages are available in your configured feeds:

```powershell
dotnet add package NDSharp
# Optional companions:
dotnet add package NDSharp.Frame
dotnet add package NDSharp.Plot
dotnet add package NDSharp.Image
dotnet add package NDSharp.Wavelets
```

To create packages from this checkout and use a local feed:

```powershell
dotnet pack src/NDSharp.slnx -c Release -o artifacts
# Run in the consumer project's directory; use your actual absolute checkout path:
dotnet nuget add source D:/Dev/2026/repos/NDSharp/artifacts --name NDSharpLocal
dotnet add package NDSharp
```

Package versions are defined individually in each `.csproj`; existing files in `artifacts/` may predate the current source. Pin versions in consumers when reproducibility matters.

## Arrays, reductions, and slicing

C# API names are case-sensitive: `np.Arange`, not Python's `np.arange`.

```csharp
using NDSharp;

var a = np.Reshape(np.Arange(0L, 6L), 2, 3);
Console.WriteLine(a.Shape[0]); // 2
Console.WriteLine(a[1, 2]);    // 5

var doubled = np.Add(a, a);
var columnTotals = np.Sum(a, axis: new[] { 0 });
var average = np.Mean(a);
Console.WriteLine(doubled);
Console.WriteLine(columnTotals);
Console.WriteLine(average);

var view = a.Get(new Slice(0, 2), new Slice(1, 3));
var independent = view.Copy();
var floating = a.AsType(DType.Float64);
```

Slice stop positions are exclusive. Basic slices can share the original buffer: use `Copy()` when independent storage is required. Reduction methods return `NDArray`, including scalar results. `GetAt(0)` retrieves the boxed value of a scalar result.

Broadcasting aligns dimensions from the right; matching lengths or a dimension of length one are compatible. Explicit dtypes are useful when integer width, floating precision, or downstream native operations matter.

## Tabular data

```csharp
using NDSharp.Frame;
using FrameIndex = NDSharp.Frame.Index;

var frame = new DataFrame(
    new[]
    {
        Column.FromLongs(new long[] { 1, 2 }),
        Column.FromStrings(new string?[] { "x", "y" })
    },
    FrameIndex.OfStrings(new[] { "id", "label" }));

Console.WriteLine(frame.NRows); // 2
Console.WriteLine(frame.GetColumn((object?)"label")[1]); // y

var series = new Series(Column.FromLongs(new long[] { 1, 22, 333 }), null, "n");
Console.WriteLine(Formatter.SeriesRepr(series, new DisplayOptions()));
```

The constructor's second argument above supplies column labels. Frame arithmetic can align by labels rather than position. Missing values can cause dtype promotion; for example, an integer column with a missing position may become floating point with `NaN`.

## Render a PNG

```csharp
using NDSharp.Plot;
using SkiaSharp;

var figure = new Figure();
var axes = figure.AddAxes(GridCell.Single(1, 1, 0));
axes.Add(new Line2D(
    new[] { 0.0, 1.0, 2.0 },
    new[] { 0.0, 1.0, 4.0 },
    SKColors.Blue));
File.WriteAllBytes("plot.png", figure.RenderPng());
```

The figure is a rendering model, not an interactive GUI window. Other artists include images and quiver arrows. `Figure.AddColorbar` requires a color-mappable artist, not a plain line.

## Image operations and native resources

```csharp
using NDSharp;
using NDSharp.Image;

var bgr = np.Full(new[] { 2, 2, 3 }, 100L, DType.UInt8);
var gray = Cv.CvtColor(bgr, 6); // OpenCV COLOR_BGR2GRAY
Console.WriteLine(gray.Shape.Length); // 2

using var mat = Mats.ToMat(gray);
var restored = Mats.FromMat(mat);
```

Images use row/column dimensions and, for color images, a channel dimension. OpenCV color images conventionally use BGR ordering. Conversion tests cover `UInt8`, `Int16`, `Float32`, and `Float64`, including non-contiguous input views. Dispose OpenCvSharp `Mat` objects with `using`.

The project references `OpenCvSharp4.runtime.win`. A successful managed build alone does not validate native loading on Linux or macOS; those deployments need a suitable OpenCvSharp native runtime and its dependencies.

## Wavelet transforms

```csharp
using NDSharp.Wavelets;

var wavelet = Wavelet.Get("haar");
var (approximation, detail) = Dwt.Dwt1D(
    new[] { 1.0, 2.0, 3.0, 4.0 }, wavelet);
var reconstructed = Dwt.Idwt1D(approximation, detail, wavelet);
Console.WriteLine(string.Join(", ", reconstructed));
```

`Wavelet.Names` lists built-in names. The default extension mode is `symmetric`; supported modes also include `zero`, `constant`, `reflect`, `periodic`, `periodization`, `antisymmetric`, `antireflect`, and `smooth`. Use the same mode for forward and inverse transforms. Output and reconstruction lengths depend on input length, filter length, and mode; do not assume an odd-length input reconstructs to exactly its original length.

## Development commands

```powershell
dotnet restore src/NDSharp.slnx
dotnet build src/NDSharp.slnx -c Release
dotnet test src/NDSharp.slnx -c Release
# Run a single suite:
dotnet test src/NDSharp.Tests/NDSharp.Tests.csproj -c Release
# Run selected tests:
dotnet test src/NDSharp.Tests/NDSharp.Tests.csproj --filter FullyQualifiedName~EigTests
```

For generated oracle data, consult [ARCHITECTURE.md](ARCHITECTURE.md). Normal builds and tests use committed fixtures.

## Troubleshooting

- **Package not found:** configure the feed containing the package or use a project reference/local package feed.
- **Native library load failure:** check OS/architecture compatibility and OpenCV or SkiaSharp native assets. Image runtime packaging is currently Windows-oriented.
- **Unexpected data changes:** a slice may share storage; make an explicit copy before mutation.
- **Unexpected dtype or missing result:** inspect dtype promotion, broadcasting shapes, and Frame label alignment.
- **Preview SDK notice:** the repository does not pin an SDK with `global.json`; an installed preview SDK may be selected even though projects target `net11.0`.

## C# Jupyter notebooks

The [notebooks](src/notebooks) are standard `.ipynb` files using the **.NET Interactive C#** kernel, not Python. All examples pin NDSharp packages to `2.5.0`. Notebook outputs are deliberately cleared so checked-in files do not contain machine-specific paths or execution state.

### Setup

**VS Code:** install Microsoft's **Polyglot Notebooks** extension and the .NET 11 SDK. Open an `.ipynb`, select its .NET Interactive C# kernel, and Run All. The extension manages the interactive host; select a host capable of loading .NET 11 assemblies. A C# kernel running on an older runtime cannot load these libraries.

**JupyterLab:** install JupyterLab using your preferred Python environment (Python is only the notebook UI, not the numerical runtime), then install/register .NET Interactive:

```powershell
dotnet tool install --global Microsoft.dotnet-interactive
dotnet interactive jupyter install
jupyter lab
```

If the tool feed only offers prereleases, select a compatible version with `--version`; kernel installation requires Jupyter to be discoverable on PATH. Select **.NET (C#)**, not Python, when opening a notebook.

The first cell restores five packages. If `2.5.0` is not available from your feeds, use the local packages in this checkout. From the repository root:

```powershell
dotnet pack src/NDSharp.slnx -c Release -o artifacts
dotnet nuget add source "$PWD/artifacts" --name NDSharp-local
```

Restart the kernel after changing feeds. NuGet sources must be visible to the account running the kernel. Alternatively, add a `#i "nuget: /absolute/path/to/artifacts"` directive before the `#r` directives. This local-feed setup loads packaged binaries; the validation command below uses current source project references, so repack after changing the libraries.

Each notebook is independent. Run setup first, then the example; restart and Run All when experimenting with variable declarations. Image examples currently need Windows native OpenCV assets; plots require compatible SkiaSharp assets. The two plotting lessons save `ndsharp-line.png` and `ndsharp-heatmap.png` in the **kernel working directory**; open these files to view the plots.

### Learning path

| Notebook | Topics |
| --- | --- |
| [01 Arrays](src/notebooks/01-arrays.ipynb) | Creation, shape, size, dtype casts |
| [02 Views](src/notebooks/02-views.ipynb) | Slicing, transpose, copy versus shared storage |
| [03 Broadcasting](src/notebooks/03-broadcasting.ipynb) | Trailing dimensions, arithmetic |
| [04 Reductions](src/notebooks/04-reductions.ipynb) | Sum, mean, median, axes and keepdims |
| [05 Linear algebra](src/notebooks/05-linear-algebra.ipynb) | Solve, determinant, floating-point tolerance |
| [06 Random](src/notebooks/06-random.ipynb) | Seeded PCG64, uniform draws, normal draws |
| [07 Signals](src/notebooks/07-signal.ipynb) | FFT, inverse FFT, frequencies, differences |
| [08 Series](src/notebooks/08-series.ipynb) | Label alignment, missing values, reductions |
| [09 CSV](src/notebooks/09-csv.ipynb) | Text parsing, inferred columns, DataFrame |
| [10 Joins](src/notebooks/10-joins.ipynb) | Outer join, missing payload promotion |
| [11 Line plot](src/notebooks/11-line-plot.ipynb) | Figure, axes, line artist, PNG |
| [12 Heatmap](src/notebooks/12-heatmap.ipynb) | Image artist, viridis, colorbar |
| [13 Images](src/notebooks/13-image.ipynb) | BGR to grayscale, Mat ownership, roundtrip |
| [14 Wavelets](src/notebooks/14-wavelets.ipynb) | Haar coefficients and reconstruction |

### Maintaining and validating samples

The authoritative lesson text and code are in `src/NDSharp.Notebooks/Program.cs`. Regenerate after editing that file rather than editing the notebook JSON or `Validation.g.cs` directly:

```powershell
dotnet run --project src/NDSharp.Notebooks
dotnet build src/NDSharp.Notebooks
dotnet run --project src/NDSharp.Notebooks -- --validate
dotnet test src/NDSharp.slnx
```

The generator writes the notebooks and a C# validation companion containing the **same example code**. Validation executes all fourteen examples, including numerical checks and native image/plot rendering. This verifies compilation and source-backed execution, not Jupyter kernel registration, NuGet restoration inside a kernel, or notebook-host display behavior. Use Run All in the actual notebook host to check those integration points. The validation helper and its xUnit tests are included in the solution but are not shipped packages. The test suite executes each lesson and checks notebook JSON, kernel metadata, and cleared outputs.

## API reference

This is a curated C# reference to the main public entry points, not an exhaustive compatibility listing. Names are case-sensitive. Python-shaped operations are exposed as C# methods; do not assume `np.linalg`, `np.random`, pandas indexers, or Python keyword spellings exist. Source links below contain exact overloads and implementations.

### NDSharp: array model and creation

Namespace: `NDSharp`. Package: `NDSharp`.

| Member | Contract / usage |
| --- | --- |
| [`NDArray`](src/NDSharp/NDArray.cs) | Typed, strided array. `new NDArray(DType.Float64, doubleBuffer, shape)` requires the CLR buffer element type to match dtype. |
| `Shape`, `Ndim`, `Size` | Dimension lengths, number of dimensions, total element count. A scalar is zero-dimensional with one element. |
| `Buffer`, `Strides`, `Offset`, `Base` | Shared CLR storage, element strides, absolute element offset, owning array for a view. Buffer order is not necessarily logical view order. |
| `GetAt(int)` | Read by logical flat position; use `GetAt(0)` to extract a scalar reduction. |
| `a[i, j]` | Coordinate indexing. |
| `a.Get(params NDIndex[] indices)` | Basic slices and advanced indexing; see [Indexing.cs](src/NDSharp/Indexing.cs). `new Slice(start, stop)` uses an exclusive stop. |
| `Copy()` | Independent logical copy. |
| `AsType(DType)` | Explicit dtype conversion. |
| `NDArray.Scalar(long/double/bool)` | Strongly typed zero-dimensional array, useful as an operand. |
| [`DType`, `DTypes`](src/NDSharp/DType.cs) | Dtype identifiers and promotion/casting helpers; inspect these rather than relying on CLR arithmetic promotion. |
| [`np.Arange`](src/NDSharp/np.Creation.cs) | Half-open range; `np.Arange(0L, 6L)` makes six integers. |
| `np.Zeros`, `np.Ones`, `np.Full` | Allocate arrays for an explicit shape; `Full(shape, value, dtype)` makes dtype intent explicit. |
| `np.Linspace` | Evenly spaced samples; inspect overload for endpoint and dtype options. |

**Storage warning:** basic slices and transpose generally share storage. Reshape can copy a non-contiguous input; `Flatten` always copies whereas `Ravel` follows reshape behavior. Do not modify shape/stride arrays or raw buffer storage unless implementing low-level interop.

### Shape, broadcasting, elementwise operations

Sources: [shape](src/NDSharp/np.Shape.cs), [elementwise](src/NDSharp/np.Elementwise.cs), [products](src/NDSharp/np.Products.cs).

| Signature / family | Meaning |
| --- | --- |
| `np.Reshape(NDArray a, params int[] newShape)` | Same element count; one inferred dimension may be `-1`. |
| `np.Transpose(NDArray a, int[]? axes = null)` | Default reverses axis order. |
| `np.ExpandDims(a, axis)`, `np.Squeeze(a, axis)` | Add/remove singleton axes. |
| `np.BroadcastTo(a, shape)` | Broadcast view; trailing dimensions must match or be one. |
| `np.Concatenate(arrays, axis = 0)` | Join along an existing axis. |
| `np.Stack(arrays, axis = 0)` | Join along a new axis. |
| `np.Add`, `Subtract`, `Multiply`, `Divide` | Elementwise arithmetic with broadcast operands. Division is not matrix solving. |
| `np.Equal`, comparison functions, `np.All`, `np.Any` | Elementwise masks and logical reductions. |
| `np.Dot`, `np.MatMul` | Products; do not substitute elementwise `Multiply` for matrix multiplication. |

### Reductions, statistics and linear algebra

Sources: [reductions](src/NDSharp/Reduce.cs), [statistics](src/NDSharp/np.Stats.cs), [linear algebra](src/NDSharp/np.LinearAlgebra.cs), [eigenvalues](src/NDSharp/np.Eig.cs).

- `np.Sum(a, axis: new[] { 1 }, keepdims: true)` reduces axis one and retains it as a singleton. Omitting axes reduces the full array. Reduction results remain `NDArray` values, including scalars.
- `np.Mean`, `np.Var`, `np.Std`, `np.Min`, `np.Max` provide basic reductions; inspect overloads for dtype and degrees-of-freedom options.
- `np.Median(NDArray a, int[]? axis = null, bool keepdims = false)` returns the median. `np.Percentile(a, q, ...)` accepts `q` as an array.
- `np.Sort(a, int? axis = -1)` and `np.ArgSort` default to the last axis; `axis: null` operates on flattened data.
- `np.Unique(a, returnIndex: ..., returnInverse: ..., returnCounts: ...)` returns a structured result; request metadata explicitly.
- `np.Diff(a, n = 1, axis = -1)` computes finite differences; `np.Convolve(a, v, mode = "full")` convolves one-dimensional data.
- `np.Solve(NDArray a, NDArray b)` solves a square linear system. `np.Inv(a)` returns an inverse; prefer Solve for solving equations.
- `np.Det(a)` returns a determinant array; `np.SlogDet(a)` returns `(sign, logdet)`.
- `np.Eigh(a)` returns `(w, v)` for symmetric/Hermitian input. `np.Eig(a)` handles general eigenproblems; check source for result representation.
- `np.Svd(a, bool fullMatrices = true)` returns an `SvdResult`; `np.SingularValues(a)` returns only singular values.
- `np.LstSq(a, b, double? rcond = null)` returns `(x, residuals, rank, s)`; `np.Pinv(a, rcond)` computes a pseudoinverse.

Use tolerance-based comparisons for floating-point results. Not all operations have identical supported ranks, dtypes, or batching rules; consult the linked implementation and tests for edge cases. Invalid dimensions/dtypes use the `NDException` family; singular systems can fail rather than produce a usable solution.

### Random and FFT

Namespaces: `NDSharp.Random` and `NDSharp`. Sources: [Generator](src/NDSharp/Generator.cs), [distributions](src/NDSharp/GeneratorDistributions.cs), [FFT](src/NDSharp/Fft.cs).

```csharp
var rng = NDSharp.Random.Generator.DefaultRng(
    new System.Numerics.BigInteger[] { 42 });
var samples = rng.StandardNormal(new[] { 100 });
var frequencyDomain = Fft.FFT(samples);
var reconstructed = Fft.IFFT(frequencyDomain);
```

- `DefaultRng(IReadOnlyList<BigInteger>? seed = null)`: null uses OS entropy; explicit seed makes runs reproducible.
- `Random(int[]? size = null)`: Float64 samples in `[0,1)`; null size means a scalar.
- `Uniform(low, high, size)` and `Normal(loc, scale, size)` have scalar and broadcast-array parameter forms.
- `Fft.FFT(a, n = null, axis = -1, norm = null)` and `IFFT` use the final axis by default. FFT is complex-valued even for real input.
- `FFT2`/`IFFT2`, `FFTN`/`IFFTN` handle multiple axes; `RFFT`/`IRFFT` handle real-transform conventions.
- `FFTFreq(n, d = 1)` returns bins for sample spacing `d`; `FFTShift` and `IFFTShift` reorder bins, not transform values.

### NDSharp.Frame: tabular model

Namespace: `NDSharp.Frame`. Package: `NDSharp.Frame`. Sources: [Column](src/NDSharp.Frame/Column.cs), [Series](src/NDSharp.Frame/Series.cs), [DataFrame](src/NDSharp.Frame/DataFrame.cs), [Index](src/NDSharp.Frame/Index.cs).

| Member | Meaning |
| --- | --- |
| `Column.FromLongs`, `FromDoubles`, `FromStrings`, `Infer` | Typed columns or inferred object values. Missing integer entries can promote to Float64. |
| `new Series(values, index, name)` | One typed column with labels; null index uses default positions. |
| `new DataFrame(data, columns, index)` | Column data, **column-name index**, then optional row index. Do not swap the two indexes. |
| `NRows`, `NCols`, `Data`, `Columns`, `Index` | Dimensions, underlying columns, column labels and row labels. |
| `GetColumn((object?)"name")` | Select a column as a Series. Explicit object cast makes label intent clear. |
| `Index.OfStrings`, `Index.Of`, `Index.Range` | String labels, arbitrary labels, positional ranges. Alias `using FIndex = NDSharp.Frame.Index;` to avoid `System.Index` ambiguity. |
| `Ops.Binary(BinOp.Add, seriesA, seriesB)` | Label-aligned arithmetic, including missing values on unmatched labels. |
| `NDSharp.Frame.Reduce.Scalar("mean", column, skipna: false)` | Scalar reduction without skipping missing values. Default skips missing values. |
| `Formatter.SeriesRepr`, `Formatter.FrameRepr` | Human-readable pandas-shaped text; plain `ToString()` is not the tabular formatter. |

Additional operations have explicit helper entry points rather than pandas-style fluent chains:

- [CSV](src/NDSharp.Frame/Csv.cs): `Csv.Read(string text, ReadOptions)` returns `(columns, names)`. Use `File.ReadAllText` before calling it for a path.
- [JSON](src/NDSharp.Frame/Json.cs) and [HTML](src/NDSharp.Frame/Html.cs): serialization/parsing helpers with explicit options.
- [Merge](src/NDSharp.Frame/Merge.cs): `Merge.Join(left, right, new Merge.Spec { LeftOn = ..., RightOn = ..., How = "outer" })`.
- [Grouping](src/NDSharp.Frame/Grouping.cs): key columns determine sorted groups; missing keys are excluded by default.
- [Reshape](src/NDSharp.Frame/Reshape.cs): stack/unstack and pivot-style operations.
- [Window](src/NDSharp.Frame/Window.cs): rolling, expanding and exponentially weighted operations.
- [TimeSeries](src/NDSharp.Frame/TimeSeries.cs), [Period](src/NDSharp.Frame/Period.cs), [Offsets](src/NDSharp.Frame/Offsets.cs): date/time, periods, resampling and offsets.

Missingness is dtype-specific; use `Column.IsNa(position)` rather than checking only for null. Labeled arithmetic can change row count and dtype even when input sizes match.

### NDSharp.Plot: rendering model

Namespace/package: `NDSharp.Plot`. Sources: [Figure](src/NDSharp.Plot/Figure.cs), [Axes](src/NDSharp.Plot/Axes.cs), [Artists](src/NDSharp.Plot/Artists.cs).

1. Create `new Figure()`; configure dimensions through `WidthIn`, `HeightIn` and related properties.
2. Add axes with `figure.AddAxes(GridCell.Single(rows, cols, index))`.
3. Attach artists using `axes.Add(...)`: `Line2D`, `ImageArtist`, `Quiver` and other explicit artist classes.
4. Use `axes.XLimSet`/`YLimSet` for explicit limits. `ImageArtist.Values` is a row-major `double[]` with `Rows`/`Cols`.
5. `Colormap.Get("viridis")` selects a registered map; `_r` names reverse maps. `Colormap.Names` lists choices.
6. `figure.AddColorbar(image, label: "value")` requires a mappable artist, not a line.
7. `figure.RenderPng()` returns PNG bytes, not a notebook display object. Save them or format them for your host.

Colors use `SkiaSharp.SKColor`/`SKColors`. Native SkiaSharp must be available on the host platform. The library is a rendering model, not an interactive plotting GUI.

### NDSharp.Image: native OpenCV boundary

Namespace/package: `NDSharp.Image`. Sources: [Mats](src/NDSharp.Image/Mats.cs), [basics](src/NDSharp.Image/Cv.Basics.cs), [filters](src/NDSharp.Image/Cv.Filters.cs), [features](src/NDSharp.Image/Cv.Features.cs), [DNN](src/NDSharp.Image/Cv.Dnn.cs).

- `Mats.ToMat(NDArray)` converts an array to an OpenCvSharp Mat. Dispose returned Mats with `using`.
- `Mats.FromMat(mat)` returns array data; roundtrip tests cover several numeric dtypes and non-contiguous input views.
- `Cv.CvtColor(image, code)` uses OpenCV integer conversion codes. `6` means BGR-to-grayscale.
- Images use height × width for grayscale and height × width × channels for color. OpenCV's conventional color order is BGR, not RGB.
- Filtering, morphology, warps and feature functions reside in the linked partial Cv sources. Native dtype/channel constraints still apply.
- DNN operations require user-supplied compatible models; the repository does not bundle pretrained networks.

The package currently supplies Windows native runtime assets. A successful managed restore does not guarantee native loading on Linux/macOS.

### NDSharp.Wavelets: discrete transforms

Namespace/package: `NDSharp.Wavelets`. Source: [Dwt.cs](src/NDSharp.Wavelets/Dwt.cs).

| Member | Contract |
| --- | --- |
| `Wavelet.Names`, `Wavelet.Get(name)` | Enumerate/get built-in filter banks; unknown names raise `NDValueException`. |
| `Name`, `FilterLength`, `Orthogonal` | Filter metadata. |
| `DecLo`, `DecHi`, `RecLo`, `RecHi` | Low/high decomposition and reconstruction coefficients. Treat these arrays as read-only filter data. |
| `Dwt.Dwt1D(double[] x, Wavelet w, string mode = "symmetric")` | Returns `(cA, cD)` approximation and detail arrays in double precision. |
| `Dwt.Idwt1D(cA, cD, w, mode)` | Reconstruct using the same wavelet and mode. |
| `Dwt.CoeffLength(inputLen, filterLen, mode)` | Coefficient length; periodization differs from ordinary extension modes. |
| `Dwt.Dwt2D`, `Dwt.Idwt2D` | Two-dimensional decomposition/reconstruction; inspect source for coefficient tuple layout. |

Supported extension modes include `zero`, `constant`, `symmetric`, `reflect`, `periodic`, `periodization`, `antisymmetric`, `antireflect`, and `smooth`. Boundary mode affects coefficients and lengths; for odd input lengths trim the reconstructed array to the original length when appropriate. The notebook validates an even-length Haar roundtrip, not every filter/mode combination.
