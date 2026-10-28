namespace NDSharp.Plot;

public static class FigureExtensions
{
    /// <summary>
    /// Adds a colorbar for <paramref name="mappable"/> (an image, contour or scatter) next to <paramref name="host"/>
    /// (default: the axes that own the mappable). Returns the colorbar; its axes are appended to the figure.
    /// </summary>
    public static Colorbar AddColorbar(this Figure fig, Artist mappable, Axes? host = null, double fraction = 0.15, double pad = 0.05, double shrink = 1, string? label = null)
    {
        if (mappable is not IMappable) throw new ArgumentException("colorbar expects a mappable (an image, contour or scatter)", nameof(mappable));
        host ??= mappable.Parent ?? throw new InvalidOperationException("the mappable is not attached to any axes");
        var cax = fig.AddAxes(null);
        cax.ColorbarHost = host;
        cax.CbFraction = fraction;
        cax.CbPad = pad;
        cax.CbShrink = shrink;
        var cb = new Colorbar(mappable, cax);
        cax.Colorbar = null;
        host.Colorbar = cb;
        var (lo, hi) = cb.Range;
        const int N = 256;
        var px = new byte[N * 4];
        for (int q = 0; q < N; q++)
        {
            // row 0 is the lowest value (origin lower)
            var col = cb.Cmap.At((q + 0.5) / N);
            px[4 * q] = col.Red; px[4 * q + 1] = col.Green; px[4 * q + 2] = col.Blue; px[4 * q + 3] = 255;
        }
        cax.Add(new ImageArtist { Rgba = px, Rows = N, Cols = 1, Origin = "lower", ExtentValue = (0, 1, lo, hi), Interpolation = "nearest" });
        cax.XLimSet = (0, 1); cax.YLimSet = (lo, hi);
        cax.YOnRight = true; cax.XTicksVisible = false; cax.XLabelsVisible = false;
        cax.BoxAspect = 20; cax.Anchor = (0, 0.5);
        if (label is not null) { cax.YLabel = label; cb.Label = label; }
        return cb;
    }
}
