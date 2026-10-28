using SkiaSharp;

namespace NDSharp.Plot;

/// <summary>
/// A field of arrows (matplotlib's <c>quiver</c>, MATLAB's <c>quiver</c>). The arrows are given in data units: each one runs from
/// (X, Y) to (X + DX, Y + DY). Choosing a scale so that arrows do not overlap is the caller's job (MATLAB and matplotlib each have their own rule).
/// The head is a filled triangle whose size is in points, shrunk on short arrows so it never exceeds the shaft.
/// </summary>
public sealed class Quiver : Artist
{
    public double[] X { get; set; }
    public double[] Y { get; set; }
    public double[] DX { get; set; }
    public double[] DY { get; set; }
    public SKColor Color { get; set; } = SKColor.Parse("#1f77b4");
    public double LineWidth { get; set; } = 1;
    /// <summary>Head length in points.</summary>
    public double HeadLength { get; set; } = 6;
    /// <summary>Head width as a fraction of its length.</summary>
    public double HeadAspect { get; set; } = 0.7;

    public Quiver(double[] x, double[] y, double[] dx, double[] dy)
    {
        if (x.Length != y.Length || x.Length != dx.Length || x.Length != dy.Length)
            throw new ArgumentException("quiver: x, y, dx and dy must have the same number of elements");
        X = x; Y = y; DX = dx; DY = dy;
        ZOrder = 2;
    }

    public override DataBounds? Bounds
    {
        get
        {
            double x0 = double.PositiveInfinity, x1 = double.NegativeInfinity, y0 = double.PositiveInfinity, y1 = double.NegativeInfinity;
            for (int i = 0; i < X.Length; i++)
            {
                foreach (var (px, py) in new[] { (X[i], Y[i]), (X[i] + DX[i], Y[i] + DY[i]) })
                {
                    if (!double.IsFinite(px) || !double.IsFinite(py)) continue;
                    x0 = Math.Min(x0, px); x1 = Math.Max(x1, px); y0 = Math.Min(y0, py); y1 = Math.Max(y1, py);
                }
            }
            return double.IsFinite(x0) ? new DataBounds(x0, x1, y0, y1) : null;
        }
    }

    public override IEnumerable<(double X, double Y)> Vertices
    {
        get
        {
            for (int i = 0; i < X.Length; i++) { yield return (X[i], Y[i]); yield return (X[i] + DX[i], Y[i] + DY[i]); }
        }
    }

    public override void Draw(DrawContext c)
    {
        var ax = c.Axes;
        var color = WithAlpha(Color, Alpha);
        using var shaft = Stroke(color, LineWidth * c.PtPx, "-", LineWidth, c.PtPx);
        using var fill = new SKPaint { Color = color, IsAntialias = true, Style = SKPaintStyle.Fill };
        for (int i = 0; i < X.Length; i++)
        {
            double x0 = ax.PxX(X[i]), y0 = ax.PxY(Y[i]), x1 = ax.PxX(X[i] + DX[i]), y1 = ax.PxY(Y[i] + DY[i]);
            if (!double.IsFinite(x0) || !double.IsFinite(y0) || !double.IsFinite(x1) || !double.IsFinite(y1)) continue;
            double dx = x1 - x0, dy = y1 - y0, len = Math.Sqrt(dx * dx + dy * dy);
            if (len < 0.5) continue;                                // a point, not an arrow
            double ux = dx / len, uy = dy / len;
            double head = Math.Min(HeadLength * c.PtPx, 0.45 * len);
            double half = head * HeadAspect / 2;
            // the shaft stops at the base of the head so the stroke does not poke through the tip
            c.Canvas.DrawLine((float)x0, (float)y0, (float)(x1 - ux * head * 0.8), (float)(y1 - uy * head * 0.8), shaft);
            using var tri = new SKPath();
            tri.MoveTo((float)x1, (float)y1);
            tri.LineTo((float)(x1 - ux * head - uy * half), (float)(y1 - uy * head + ux * half));
            tri.LineTo((float)(x1 - ux * head + uy * half), (float)(y1 - uy * head - ux * half));
            tri.Close();
            c.Canvas.DrawPath(tri, fill);
        }
    }

    public override void DrawLegendHandle(DrawContext c, SKRect box)
    {
        using var paint = Stroke(WithAlpha(Color, Alpha), LineWidth * c.PtPx, "-", LineWidth, c.PtPx);
        c.Canvas.DrawLine(box.Left, box.MidY, box.Right, box.MidY, paint);
    }
}


