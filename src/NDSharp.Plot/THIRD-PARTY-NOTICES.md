# Third-party notices — NDSharp.Plot

NDSharp.Plot itself is MIT-licensed (see `LICENSE`, Copyright (c) 2026 Marco Parenzan).
It incorporates, adapts or depends on the third-party material listed below. Each
component keeps its own license; the notices required by those licenses follow.

Source of the texts below: the license files shipped with matplotlib 3.11.2
(`LICENSE`, `LICENSE_COLORBREWER`, `LICENSE_YORICK`), the headers of the upstream
colormap files, and the `LICENSE_DEJAVU` file packed with this package.

| Component | Used for | License |
|---|---|---|
| matplotlib | Behaviour, algorithms and colormap data adapted or generated from matplotlib 3.11.2 | matplotlib License (PSF-style) |
| ColorBrewer | 22 colormaps | Apache License 2.0 (Apache-style notice) |
| Yorick / Gist | `gist_gray` colormap | BSD-style, University of California |
| Wistia heatmap palette | `Wistia` colormap | MIT |
| viridis, plasma, magma, inferno | Colormaps | CC0 1.0 (public domain dedication) |
| turbo | Colormap | Apache License 2.0 (Google LLC) |
| DejaVu Sans | Embedded fonts | Bitstream Vera / Arev / public domain (see `LICENSE_DEJAVU`) |
| SkiaSharp | Rasterization | MIT (notices travel with the SkiaSharp package) |

## 1. matplotlib

### What was taken, and what was changed

NDSharp.Plot reproduces the model and numeric behaviour of matplotlib 3.11.2
(Figure / Axes / artists). This is the brief summary of changes required by
clause 3 of the matplotlib License. The work is a C# reimplementation, not a
redistribution of the Python source:

- `Colormaps.g.cs`: look-up tables generated from matplotlib 3.11.2 with
  `tools/oracle/gen_colormaps.py`.
- `Ticks.cs`: adapted from `MaxNLocator`, `ScalarFormatter` and tick-space rules.
- `Figure.cs`: adapted from `tight_layout` and `make_axes`.
- `Artists.cs`: `FancyArrow`, markers, patch and collection rules.
- `Plot3D.cs`: the `mplot3d` camera model.
- `Axes.cs`, `Text.cs`, `Contour.cs`: axis-expansion rules, text metrics and band
  coloring aligned with matplotlib.

Changes: ported from Python to C#, rendered with SkiaSharp instead of Agg, reduced
to the subset used by NDSharp, and extended with NDSharp-specific behaviour.

### License text (matplotlib versions 1.3.0 and later)

```text
License agreement for matplotlib versions 1.3.0 and later
=========================================================

1. This LICENSE AGREEMENT is between the Matplotlib Development Team
("MDT"), and the Individual or Organization ("Licensee") accessing and
otherwise using matplotlib software in source or binary form and its
associated documentation.

2. Subject to the terms and conditions of this License Agreement, MDT
hereby grants Licensee a nonexclusive, royalty-free, world-wide license
to reproduce, analyze, test, perform and/or display publicly, prepare
derivative works, distribute, and otherwise use matplotlib
alone or in any derivative version, provided, however, that MDT's
License Agreement and MDT's notice of copyright, i.e., "Copyright (c)
2012- Matplotlib Development Team; All Rights Reserved" are retained in
matplotlib  alone or in any derivative version prepared by
Licensee.

3. In the event Licensee prepares a derivative work that is based on or
incorporates matplotlib or any part thereof, and wants to
make the derivative work available to others as provided herein, then
Licensee hereby agrees to include in any such work a brief summary of
the changes made to matplotlib .

4. MDT is making matplotlib available to Licensee on an "AS
IS" basis.  MDT MAKES NO REPRESENTATIONS OR WARRANTIES, EXPRESS OR
IMPLIED.  BY WAY OF EXAMPLE, BUT NOT LIMITATION, MDT MAKES NO AND
DISCLAIMS ANY REPRESENTATION OR WARRANTY OF MERCHANTABILITY OR FITNESS
FOR ANY PARTICULAR PURPOSE OR THAT THE USE OF MATPLOTLIB
WILL NOT INFRINGE ANY THIRD PARTY RIGHTS.

5. MDT SHALL NOT BE LIABLE TO LICENSEE OR ANY OTHER USERS OF MATPLOTLIB
 FOR ANY INCIDENTAL, SPECIAL, OR CONSEQUENTIAL DAMAGES OR
LOSS AS A RESULT OF MODIFYING, DISTRIBUTING, OR OTHERWISE USING
MATPLOTLIB , OR ANY DERIVATIVE THEREOF, EVEN IF ADVISED OF
THE POSSIBILITY THEREOF.

6. This License Agreement will automatically terminate upon a material
breach of its terms and conditions.

7. Nothing in this License Agreement shall be deemed to create any
relationship of agency, partnership, or joint venture between MDT and
Licensee.  This License Agreement does not grant permission to use MDT
trademarks or trade name in a trademark sense to endorse or promote
products or services of Licensee, or any third party.

8. By copying, installing or otherwise using matplotlib ,
Licensee agrees to be bound by the terms and conditions of this License
Agreement.
```

## 2. ColorBrewer color schemes

Blues, Reds, Greens, Oranges, Purples, Greys, YlOrRd, YlGnBu, RdBu, RdYlBu, RdYlGn,
Spectral, PiYG, PRGn, BrBG, Set1, Set2, Set3, Paired, Pastel1, Dark2, Accent.
Developed by Cynthia Brewer (https://colorbrewer2.org/), included in matplotlib
under the following license.

```text
Apache-Style Software License for ColorBrewer software and ColorBrewer Color Schemes

Copyright (c) 2002 Cynthia Brewer, Mark Harrower, and The Pennsylvania State University.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
```

## 3. Yorick / Gist colormap (`gist_gray`)

```text
BSD-style license for gist/yorick colormaps.

Copyright:

  Copyright (c) 1996.  The Regents of the University of California.
			 All rights reserved.

Permission to use, copy, modify, and distribute this software for any
purpose without fee is hereby granted, provided that this entire
notice is included in all copies of any software which is or includes
a copy or modification of this software and in all copies of the
supporting documentation for such software.

This work was produced at the University of California, Lawrence
Livermore National Laboratory under contract no. W-7405-ENG-48 between
the U.S. Department of Energy and The Regents of the University of
California for the operation of UC LLNL.


			      DISCLAIMER

This software was prepared as an account of work sponsored by an
agency of the United States Government.  Neither the United States
Government nor the University of California nor any of their
employees, makes any warranty, express or implied, or assumes any
liability or responsibility for the accuracy, completeness, or
usefulness of any information, apparatus, product, or process
disclosed, or represents that its use would not infringe
privately-owned rights.  Reference herein to any specific commercial
products, process, or service by trade name, trademark, manufacturer,
or otherwise, does not necessarily constitute or imply its
endorsement, recommendation, or favoring by the United States
Government or the University of California.  The views and opinions of
authors expressed herein do not necessarily state or reflect those of
the United States Government or the University of California, and
shall not be used for advertising or product endorsement purposes.


				AUTHOR

David H. Munro wrote Yorick and Gist.  Berkeley Yacc (byacc) generated
the Yorick parser.  The routines in Math are from LAPACK and FFTPACK;
MathC contains C translations by David H. Munro.  The algorithms for
Yorick's random number generator and several special functions in
Yorick/include were taken from Numerical Recipes by Press, et. al.,
although the Yorick implementations are unrelated to those in
Numerical Recipes.  A small amount of code in Gist was adapted from
the X11R4 release, copyright M.I.T. -- the complete copyright notice
may be found in the (unused) file Gist/host.c.
```

## 4. Wistia heatmap palette (`Wistia`)

Source: https://github.com/wistia/heatmap-palette

```text
The MIT License (MIT)

Copyright (c) 2014 jringenberg

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 5. viridis, plasma, magma, inferno

Colormaps by Nathaniel J. Smith, Stefan van der Walt and (viridis) Eric Firing.
Upstream notice (https://github.com/BIDS/colormap):

> This file and the colormaps in it are released under the CC0 license /
> public domain dedication. We would appreciate credit if you use or
> redistribute these colormaps, but do not impose any legal restrictions.

## 6. turbo

Look-up table by Anton Mikhailov (Google LLC). Upstream header:
`Copyright 2019 Google LLC. SPDX-License-Identifier: Apache-2.0`
(https://gist.github.com/mikhailov-work/6a308c20e494d9e0ccc29036b28faa7a).
The Apache License 2.0 is available at http://www.apache.org/licenses/LICENSE-2.0.

## 7. Items not verified

These were checked against nothing beyond matplotlib's own sources and are listed
without a claimed license:

- `cividis` (Nuñez, Anderton, Renslow) and `twilight` (Bastian Bechtold): the
  upstream license was not checked.
- The remaining maps (`gray`, `jet`, `hot`, `hsv`, `rainbow`, `coolwarm`, `bwr`,
  `seismic`, `bone`, `copper`, `cool`, `spring`, `summer`, `autumn`, `winter`,
  `binary`, `afmhot`, `nipy_spectral`, `terrain`, `ocean`, `gnuplot`, `flag`,
  `prism`, `tab10`, `tab20`) are defined by matplotlib and covered by its license
  as shown in section 1; their individual upstream origins were not checked.

## 8. Fonts

DejaVu Sans, Bold and Oblique are embedded. Their license is in the file
`LICENSE_DEJAVU` at the root of this package.

## 9. SkiaSharp

SkiaSharp 3.119.0 (MIT) and its native assets are referenced as NuGet dependencies.
Their own `LICENSE.txt` and `THIRD-PARTY-NOTICES.txt` are inside those packages.
