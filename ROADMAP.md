# NDSharp — Roadmap

NDSharp had no roadmap of its own; its plan lived in the suite roadmap. This is the project view. The suite plan is
[../ScienceSharp/ROADMAP.md](../ScienceSharp/ROADMAP.md); the older consolidation plan is [../ScienceSharp/docs/consolidation/ROADMAP.md](../ScienceSharp/docs/consolidation/ROADMAP.md).

## State (2026-10-30)

- Libraries: `NDSharp`, `.Frame`, `.Plot`, `.Image`, `.Wavelets`, all `net11.0`, packed in the local feed at 2.5.x–2.8.x `-net11-preview`.
- `NDSharp.Plot` consumes PlotSharp for linear contours, the physical field sampler and colour-band selection.
- Gate: green. Contour notebook runs in JupyterSharp through the PySharp kernel.
- MIT; `THIRD-PARTY-NOTICES.md` packed with `NDSharp.Plot`.

## Language common core — step 0 (partial, source branch)

Typed `ArrayView<T>` and zero-copy NDArray round-trip adapters are implemented.
Metadata is snapshotted and reachable storage validated. MatSharp/PySharp adapters,
their baseline suites and the complete ecosystem gate remain pending. No package
was produced or published; source version is 2.8.1-net11-preview.2.

## Next

1. **Define the perimeter** (R03): what stays in NDSharp.Plot and what moves to PlotSharp.
2. **Close contours in that perimeter** (R07): automatic levels, repeated levels, palette/alpha/raster documented as local.
3. **Run the other notebooks that use the migrated APIs** (R08). All 14 notebooks in `src/notebooks` run in the gate (their package pins moved to the current feed versions on 2026-10-30, OPENPOINTS N-12).
4. **Decide the other chart categories** (R14).
5. **Raster backend on Linux and macOS** (R15, D-F2).
6. **Licensing items** for the package: cividis/twilight, Apache text, OpenCvSharp4 notices.
7. **Documentation:** turn ARCHITECTURE.md and USAGE.md from validation logs into an architecture and API guide.
8. **Publication** (D-F3): republish `NDSharp.Plot` with the PlotSharp dependency, only when asked.

## Not planned

- A Python-compatible runtime (that is PySharp).
- A tool that evaluates NumPy-style expressions (that is `pysharp`). The `ndsharp` tool only describes and converts tables.

Open questions: [OPENPOINTS.md](OPENPOINTS.md).

## Geometry phase 5 — camera and painter ordering
NDSharp.Plot delegates perspective projection, world/view/projection matrices, aspect normalization, projected-box fit, stable back-to-front sorting and average depth to Camera<T>/DepthOrdering in MathemaSharp.Geometry. Public Plot3D.Project stays unchanged. Artists, Skia paths/paints, frame panes, grid/ticks/text and viewport acquisition remain in NDSharp. Inspected Plot3D.cs and Artists.cs: no existing orthographic projection or 3-D clipping to migrate; the latter contains separate 2-D annotation clipping.
Fingerprint on 2304 projected points plus scatter/polygon order and ties, including inverted limits, viewport sizes and near-pole/pole views: 5B92443FDF5A7BBD506E3693B9C8F67E05CAE66D822B3A65ADDEBF48324ABF7C. Two baseline runs, post-route run and post-removal engine-order run identical. Original and engine-order harnesses in BuildArtifacts/tmp. Legacy arithmetic retained (including IEEE results for singular views/ranges); no new validation or clipping semantics. Generic constants round from the predecessor's double literals.
Geometry 39/39 across four formats; MathemaSharp solution 896 passed, zero warnings/errors in build. NDSharp solution 862 passed (core 700, Plot 46), no skipped/failed; build 15 nullable warnings in unchanged Image/Frame files. All runs logged through Log-TestRun.ps1; details in BuildArtifacts/logs/geometry-contours.log. Full pre-change NDSharp suite was not run; migration baseline is the repeated fingerprint. Geometry preview.8, NDSharp.Plot preview.3; no packages/push/install. Source phases 1–5 are implemented in inspected scope; phase 6 excluded. Full ecosystem gate, prebuilt consumers, independent matplotlib accuracy, performance and broad conditioning remain unverified.
