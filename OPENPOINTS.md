# NDSharp — Open points

Suite-wide points are in [../ScienceSharp/OPENPOINTS.md](../ScienceSharp/OPENPOINTS.md).

| # | Point | Status |
| --- | --- | --- |
| N-1 | **Perimeter of NDSharp** (suite R03): which geometries belong to the core and which Figure/Axes/Skia capabilities stay in the consuming backend. | open |
| N-2 | **Other chart categories** (R14): grouped/stacked bars, histograms, advanced quiver, layout/tick/font, artists, 3D — keep, adapt to PlotSharp, or exclude explicitly. | open |
| N-3 | Palette, alpha and the native raster/Skia renderer of NDSharp.Plot are still local to NDSharp; only linear contours, the physical sampler and band selection come from PlotSharp. | known |
| N-4 | **Republishing `NDSharp.Plot` with its PlotSharp dependency** (D-F3 / N4 / R12): decided "not now". Nothing of NDSharp is on nuget.org (checked 2026-10-30), so every package is local anyway. The notebook pins, which used to stay at 2.5.0 until this was decided, moved to the current local versions on 2026-10-30 (see N-12). | deferred |
| N-5 | The upstream licence of `cividis` and `twilight` is not verified; Apache-2.0 text only by URL; OpenCvSharp4 notices missing in `NDSharp.Image`. See [../ScienceSharp/LICENSING.md](../ScienceSharp/LICENSING.md). | open |
| N-6 | CLI `ndsharp` exists (info, head, describe, convert; CSV in, JSON or CSV out). It does not read NPY, Parquet or Excel, and does not plot. | done, extend later (suite G9) |
| N-7 | Raster backend (SkiaSharp natives, fonts, DPI, alpha) is verified on Windows x64 only (R15). | open |
| N-8 | All 14 notebooks in `src/notebooks` are a gate step (`ndsharp-notebooks`, runner `MathemaSharp/ops/RunNDSharpNotebooks.cs`, about 30 s): each must exit 0 with no error output and at least one output; the runner's exclusion list is empty. Other example notebooks that use the migrated APIs (R08) are not covered. | done for `src/notebooks` |
| N-9 | `ARCHITECTURE.md` and `USAGE.md` are mostly a log of validation steps ("Current complete Science baseline", "Band selection…"). They should be reorganised into a description of the architecture and of the API. | open |
| N-10 | `PrintOptions.Default` is a single shared object (a concurrency risk for hosts running several engines). Not used by MatSharp; PySharp status not checked. | open |
| N-11 | Contour levels and axis behaviour differ from matplotlib in places; the author chose to keep them and declare the differences (PlotSharp CONTRATTI §7, option c). | decided |
| N-12 | The 14 notebooks in `src/notebooks` run in JupyterSharp (`#r "nuget:"`, JupyterSharp `preview.4`) and are a gate step (N-8). **They pinned the old local 2.5.0 packages, and `12-heatmap` failed on them** (`Figure.AddColorbar` is newer than 2.5.0). On 2026-10-30 the pins moved to the current feed versions (NDSharp 2.8.1, Frame 2.5.1, Plot 2.6.3, Image 2.5.1, Wavelets 2.5.1, all `-net11-preview`), set once in `src/NDSharp.Notebooks/Program.cs` (`PackageVersions`) and regenerated; all 14 pass. When the feed moves on, change those five constants, regenerate (build the generator with `-p:UseArtifactsOutput=false` so it can find the repo) and run the gate. | done |
