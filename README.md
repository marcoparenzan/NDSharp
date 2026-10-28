# NDSharp

NDSharp is a native .NET n-dimensional array library with NumPy-style semantics, accompanied by libraries for tabular data, plotting, image processing, and wavelets. It requires **no Python runtime**. The libraries were extracted from PySharp so interpreters and other .NET applications can share one numerical implementation.

This repository contains libraries, not a standalone application or command-line tool. All projects target **.NET 10**.

## Libraries

| Package | Purpose | Main entry points |
| --- | --- | --- |
| `NDSharp` | Typed arrays, strided views, broadcasting, elementwise operations, reductions, linear algebra, FFT, PCG64 random generation | `NDArray`, `np` |
| `NDSharp.Frame` | pandas-shaped columns, Series, DataFrame, indexes, time series, grouping, windows, data I/O | `Column`, `Series`, `DataFrame` |
| `NDSharp.Plot` | matplotlib-shaped plotting model and PNG rendering with SkiaSharp | `Figure`, `Axes`, artists |
| `NDSharp.Image` | OpenCV-backed image operations on NDSharp arrays | `Cv`, `Mats` |
| `NDSharp.Wavelets` | Double-precision discrete wavelet transforms using PyWavelets filter banks | `Wavelet`, `Dwt` |

These are C# APIs inspired by the corresponding Python libraries, not a promise of complete Python API compatibility. Language bindings live in separate repositories, including PySharp and MatSharp.

## Documentation

- [USAGE.md](USAGE.md): installation, C# examples, [notebook guide](USAGE.md#c-jupyter-notebooks), [API reference](USAGE.md#api-reference), build/test/package commands, troubleshooting.
- [C# Jupyter notebooks](src/notebooks): 14 self-contained lessons, from arrays and broadcasting to data frames, PNG plots, OpenCV, and wavelets.
- [ARCHITECTURE.md](ARCHITECTURE.md): repository layout, array storage, companion libraries, tests, and generated data.

## Quick start

Install the .NET 10 SDK, then create a consumer project:

```powershell
dotnet new console -n ArrayDemo --framework net10.0
cd ArrayDemo
dotnet add package NDSharp
```

Package availability depends on your configured NuGet feeds. To consume a checkout directly, use a project reference instead; see [USAGE.md](USAGE.md).

Replace `Program.cs` with:

```csharp
using NDSharp;

var matrix = np.Reshape(np.Arange(0L, 6L), 2, 3);
Console.WriteLine(matrix);
Console.WriteLine(np.Sum(matrix));
```

Run with `dotnet run`.

## Build and test this repository

From the repository root:

```powershell
dotnet build src/NDSharp.slnx -c Release
dotnet test src/NDSharp.slnx -c Release
```

Image operations currently include a **Windows OpenCV native runtime** dependency. Other operating systems require an appropriate native OpenCV deployment; see the usage guide. Plotting also uses native SkiaSharp assets.

Python is only needed to regenerate optional oracle data; the generated files are committed and normal builds do not need Python.

## License

The project uses the [MIT license](LICENSE). Plotting bundles DejaVu Sans fonts; their separate license is in [src/NDSharp.Plot/Fonts/LICENSE_DEJAVU](src/NDSharp.Plot/Fonts/LICENSE_DEJAVU).
