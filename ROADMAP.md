# NDSharp — Roadmap

NDSharp had no roadmap of its own; its plan lived in the suite roadmap. This is the project view. The suite plan is
[../ScienceSharp/ROADMAP.md](../ScienceSharp/ROADMAP.md); the older consolidation plan is [../ROADMAP.md](../ROADMAP.md).

## State (2026-10-30)

- Libraries: `NDSharp`, `.Frame`, `.Plot`, `.Image`, `.Wavelets`, all `net11.0`, packed in the local feed at 2.5.x–2.8.x `-net11-preview`.
- `NDSharp.Plot` consumes PlotSharp for linear contours, the physical field sampler and colour-band selection.
- Gate: green. Contour notebook runs in JupyterSharp through the PySharp kernel.
- MIT; `THIRD-PARTY-NOTICES.md` packed with `NDSharp.Plot`.

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
