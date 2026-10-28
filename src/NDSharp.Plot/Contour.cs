// Copyright (c) 2026 Marco Parenzan
//
// Licensed under the MIT License. See the LICENSE file in the project
// root for full license information.

using SkiaSharp;

namespace NDSharp.Plot;

/// <summary>Anything a colorbar can describe: a colormap and the value range it spans.</summary>
public interface IMappable
{
    Colormap Cmap { get; }
    (double lo, double hi) Range { get; }
}

/// <summary>contour / contourf over a rectilinear grid: marching squares for lines, per-pixel band lookup for filled regions.</summary>
public sealed class ContourArtist : Artist, IMappable
{
    public double[] Xs { get; set; } = Array.Empty<double>();
    public double[] Ys { get; set; } = Array.Empty<double>();
    /// <summary>Row-major values, Ys.Length rows × Xs.Length columns.</summary>
    public double[] Z { get; set; } = Array.Empty<double>();
    public double[] Levels { get; set; } = Array.Empty<double>();
    public bool Filled { get; set; }
    public Colormap Cmap { get; set; } = Colormap.Get("viridis");
    public SKColor[]? Colors { get; set; }
    public double LineWidth { get; set; } = 1.5;
    public string? LineStyle { get; set; }

    public (double lo, double hi) Range => (Levels.Min(), Levels.Max());

    public ContourArtist() { ZOrder = 1; }

    public override DataBounds? Bounds => new(Xs.Min(), Xs.Max(), Ys.Min(), Ys.Max());
    public override (double? X0, double? X1, double? Y0, double? Y1) Sticky => (Xs.Min(), Xs.Max(), Ys.Min(), Ys.Max());

    private SKColor LevelColor(int i, int n)
    {
        if (Colors is { Length: > 0 } c) return c[i % c.Length];
        var (lo, hi) = Range;
        return Cmap.At(hi == lo ? 0 : (Levels[i] - lo) / (hi - lo));
    }

    public static double[] AutoLevels(double zmin, double zmax, int n)
    {
        var ticks = Ticks.Locate(zmin, zmax, n + 1, fineSteps: true);
        return ticks;
    }

    public override void Draw(DrawContext c)
    {
        if (Xs.Length < 2 || Ys.Length < 2) return;
        if (Filled) DrawFilled(c); else DrawLines(c);
    }



    private void DrawLines(DrawContext c)
    {
        var ax = c.Axes;
        bool mono = Colors is { Length: 1 };
        double zlo = Z.Where(double.IsFinite).Min(), zhi = Z.Where(double.IsFinite).Max();
        for (int li = 0; li < Levels.Length; li++)
        {
            double lv = Levels[li];
            if (lv <= zlo || lv >= zhi) continue;
            using var path = new SKPath();
            // Shared geometry preserves physical/nonuniform edges; NDSharp keeps levels, paint and Skia projection.
            foreach (var segment in PlotSharp.RectilinearContours.Trace(Xs, Ys, Z, lv))
                Seg(path, ax, (segment.X1, segment.Y1), (segment.X2, segment.Y2));
            string style = LineStyle ?? (mono && lv < 0 ? "--" : "-");
            using var paint = Stroke(WithAlpha(LevelColor(li, Levels.Length), Alpha), LineWidth * c.PtPx, style, LineWidth, c.PtPx);
            c.Canvas.DrawPath(path, paint);
        }
    }

    private static void Seg(SKPath p, Axes ax, (double x, double y) a, (double x, double y) b)
    {
        p.MoveTo((float)ax.PxX(a.x), (float)ax.PxY(a.y));
        p.LineTo((float)ax.PxX(b.x), (float)ax.PxY(b.y));
    }


    private void DrawFilled(DrawContext c)
    {
        var ax = c.Axes;
        var sampler = new PlotSharp.RectilinearFieldSampler(Xs, Ys, Z);
        var bands = new PlotSharp.ContourBands(Levels);
        var box = ax.Box;
        var (vx0, vx1) = ax.ViewX(); var (vy0, vy1) = ax.ViewY();
        int w = Math.Max(1, (int)Math.Ceiling(box.Width)), h = Math.Max(1, (int)Math.Ceiling(box.Height));
        using var bmp = new SKBitmap(new SKImageInfo(w, h, SKColorType.Rgba8888, SKAlphaType.Unpremul));
        var px = new byte[w * h * 4];
        double xmin = Xs.Min(), xmax = Xs.Max(), ymin = Ys.Min(), ymax = Ys.Max();
        for (int py = 0; py < h; py++)
            for (int pxl = 0; pxl < w; pxl++)
            {
                double x = vx0 + (pxl + 0.5) / box.Width * (vx1 - vx0), y = vy0 + (1 - (py + 0.5) / box.Height) * (vy1 - vy0);
                if (x < xmin || x > xmax || y < ymin || y > ymax) continue;
                double z = sampler.Sample(x, y);
                int band = bands.Find(z);
                if (band < 0) continue;
                // matplotlib colours band k by the midpoint of its bounds, normalised over the full level range
                double normalized = bands.NormalizedMidpoint(band);
                var col = Colors is { Length: > 0 } cs ? cs[band % cs.Length] : Cmap.At(normalized);
                int o = 4 * (py * w + pxl);
                px[o] = col.Red; px[o + 1] = col.Green; px[o + 2] = col.Blue; px[o + 3] = (byte)(255 * Alpha);
            }
        System.Runtime.InteropServices.Marshal.Copy(px, 0, bmp.GetPixels(), px.Length);
        using var img = SKImage.FromBitmap(bmp);
        using var paint = new SKPaint { IsAntialias = false };
        c.Canvas.DrawImage(img, new SKRect(0, 0, w, h), new SKRect(box.Left, box.Top, box.Left + w, box.Top + h), new SKSamplingOptions(SKFilterMode.Nearest), paint);
    }
}
