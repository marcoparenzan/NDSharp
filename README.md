<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="branding/logo-dark.svg">
    <img alt="NDSharp" src="branding/logo.svg" height="72">
  </picture>
</p>

# NDSharp

> **Part of [ScienceSharp](../ScienceSharp)** — the Science suite. Targets **.NET 11**. Status: preview, not published. License: MIT.
> **Form:** NuGet libraries (`NDSharp`, `NDSharp.Frame`, `NDSharp.Plot`, `NDSharp.Image`, `NDSharp.Wavelets`). Global tool `ndsharp` (describe, head, info, convert CSV and JSON).

| [README](README.md) | [USAGE](USAGE.md) (install first) | [ARCHITECTURE](ARCHITECTURE.md) | [ROADMAP](ROADMAP.md) | [OPENPOINTS](OPENPOINTS.md) |
| --- | --- | --- | --- | --- |

NDSharp is a native .NET n-dimensional array library with NumPy-style semantics, accompanied by libraries for tabular data, plotting, image processing, and wavelets. It requires **no Python runtime**. The libraries were extracted from PySharp so interpreters and other .NET applications can share one numerical implementation.

This repository contains libraries and one command-line tool, `ndsharp`. All projects target **.NET 11**.

## Typed array views (source branch)

`ArrayView<T>` provides zero-copy typed storage with snapshot shape/stride metadata.
`NDArray.AsArrayView<T>()` and `NDArray.FromArrayView(view, dtype)` preserve storage,
strides and offset. Language adapters are not implemented yet; this is only the
NDSharp portion of language-common-core step 0. No package has been published.

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

Install the .NET 11 SDK, then create a consumer project:

```powershell
dotnet new console -n ArrayDemo --framework net11.0
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

## AI assistance

This code was developed with the assistance of generative AI tools. The matplotlib-derived parts are credited in [src/NDSharp.Plot/THIRD-PARTY-NOTICES.md](src/NDSharp.Plot/THIRD-PARTY-NOTICES.md).

## A note on licensing

This is a development exercise, not a commercial effort and not a way to take anything from anyone. It tries to show that .NET and C# can do the same kind of
scientific work as Python, in the hope that someone will try it. Its licensing, including the third-party material it adapts or credits, was reviewed with the help
of generative AI, with the aim of the greatest respect for the rights of the owners. That is not legal advice, and some points are still open: they are listed in
[LICENSING.md](../ScienceSharp/LICENSING.md). If you own material that is used here and have a concern, please open an issue: it will be discussed and, if needed, fixed or removed.

## Geometry phase 5 — camera and painter ordering
NDSharp.Plot delegates perspective projection, world/view/projection matrices, aspect normalization, projected-box fit, stable back-to-front sorting and average depth to Camera<T>/DepthOrdering in MathemaSharp.Geometry. Public Plot3D.Project stays unchanged. Artists, Skia paths/paints, frame panes, grid/ticks/text and viewport acquisition remain in NDSharp. Inspected Plot3D.cs and Artists.cs: no existing orthographic projection or 3-D clipping to migrate; the latter contains separate 2-D annotation clipping.
Fingerprint on 2304 projected points plus scatter/polygon order and ties, including inverted limits, viewport sizes and near-pole/pole views: 5B92443FDF5A7BBD506E3693B9C8F67E05CAE66D822B3A65ADDEBF48324ABF7C. Two baseline runs, post-route run and post-removal engine-order run identical. Original and engine-order harnesses in BuildArtifacts/tmp. Legacy arithmetic retained (including IEEE results for singular views/ranges); no new validation or clipping semantics. Generic constants round from the predecessor's double literals.
Geometry 39/39 across four formats; MathemaSharp solution 896 passed, zero warnings/errors in build. NDSharp solution 862 passed (core 700, Plot 46), no skipped/failed; build 15 nullable warnings in unchanged Image/Frame files. All runs logged through Log-TestRun.ps1; details in BuildArtifacts/logs/geometry-contours.log. Full pre-change NDSharp suite was not run; migration baseline is the repeated fingerprint. Geometry preview.8, NDSharp.Plot preview.3; no packages/push/install. Source phases 1–5 are implemented in inspected scope; phase 6 excluded. Full ecosystem gate, prebuilt consumers, independent matplotlib accuracy, performance and broad conditioning remain unverified.
