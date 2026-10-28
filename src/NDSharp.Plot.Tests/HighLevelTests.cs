using SkiaSharp;

namespace NDSharp.Plot.Tests;

public class HighLevelTests
{
    static SKBitmap Decode(byte[] png) => SKBitmap.Decode(png);

    static int CountPixels(SKBitmap bmp, Func<SKColor, bool> pred)
    {
        int n = 0;
        for (int y = 0; y < bmp.Height; y++) for (int x = 0; x < bmp.Width; x++) if (pred(bmp.GetPixel(x, y))) n++;
        return n;
    }

    [Fact]
    public void QuiverDrawsArrowsInItsColour()
    {
        var fig = new Figure();
        var ax = fig.AddAxes(GridCell.Single(1, 1, 0));
        var red = new SKColor(255, 0, 0);
        double[] x = { 0, 1, 2 }, y = { 0, 0, 0 }, u = { 1, 1, 1 }, v = { 0, 0, 0 };
        ax.Add(new Quiver(x, y, u, v) { Color = red, LineWidth = 2 });
        var png = fig.RenderPng();
        Assert.Equal(0x89, png[0]);                                   // PNG signature
        using var bmp = Decode(png);
        Assert.True(CountPixels(bmp, c => c.Red > 200 && c.Green < 60 && c.Blue < 60) > 100, "arrows must leave red pixels");
    }

    [Fact]
    public void QuiverBoundsCoverTailsAndTips()
    {
        var q = new Quiver(new[] { 0.0, 1 }, new[] { 0.0, 1 }, new[] { 2.0, 0 }, new[] { 0.0, -3 });
        var b = q.Bounds!.Value;
        Assert.Equal((0.0, 2.0, -2.0, 1.0), (b.X0, b.X1, b.Y0, b.Y1));
    }

    [Fact]
    public void QuiverRejectsMismatchedLengths()
        => Assert.Throws<ArgumentException>(() => new Quiver(new[] { 0.0 }, new[] { 0.0, 1 }, new[] { 1.0 }, new[] { 1.0 }));

    [Fact]
    public void AddColorbarAttachesAColorbarAxesAndRenders()
    {
        var fig = new Figure();
        var ax = fig.AddAxes(GridCell.Single(1, 1, 0));
        var img = new ImageArtist { Rows = 2, Cols = 2, Values = new[] { 0.0, 1, 2, 3 }, Cmap = Colormap.Get("viridis") };
        ax.Add(img);
        var cb = fig.AddColorbar(img, label: "value");
        Assert.Same(cb, ax.Colorbar);
        Assert.Equal((0.0, 3.0), cb.Range);
        Assert.Equal(2, fig.Axes.Count);
        Assert.True(fig.Axes[1].HasColorbarAxes);
        using var bmp = Decode(fig.RenderPng());
        // viridis starts dark purple and ends yellow: both must be on the canvas (image + bar)
        Assert.True(CountPixels(bmp, c => c.Red > 230 && c.Green > 200 && c.Blue < 80) > 50);
        Assert.True(CountPixels(bmp, c => c.Red < 90 && c.Green < 20 && c.Blue > 60) > 50);
    }

    [Fact]
    public void AddColorbarRejectsNonMappables()
    {
        var fig = new Figure();
        var ax = fig.AddAxes(GridCell.Single(1, 1, 0));
        var line = new Line2D(new[] { 0.0, 1 }, new[] { 0.0, 1 }, SKColors.Black);
        ax.Add(line);
        Assert.Throws<ArgumentException>(() => fig.AddColorbar(line));
    }

    static bool DarkPixelNear(SKBitmap bmp, double cx, double cy, int radius = 1)
    {
        for (int y = (int)Math.Round(cy) - radius; y <= (int)Math.Round(cy) + radius; y++)
            for (int x = (int)Math.Round(cx) - radius; x <= (int)Math.Round(cx) + radius; x++)
                if (x >= 0 && y >= 0 && x < bmp.Width && y < bmp.Height && bmp.GetPixel(x, y) is { Red: < 100, Green: < 100, Blue: < 100 }) return true;
        return false;
    }

    [Theory]
    [InlineData(true, 5.6, 4.2)]
    [InlineData(false, 5.6, 4.2)]
    [InlineData(true, 6.4, 4.8)]
    [InlineData(false, 6.4, 4.8)]
    [InlineData(true, 8, 6)]
    [InlineData(false, 3.3, 2.7)]
    [InlineData(true, 7.1, 3.9)]
    public void TickExactlyOnTheLimitIsDrawn(bool equalAspect, double wIn, double hIn)
    {
        // limits (0, 1) put a tick at 1.0 right on the axes edge; float/double rounding must not make it vanish
        var fig = new Figure { WidthIn = wIn, HeightIn = hIn };
        var ax = fig.AddAxes(GridCell.Single(1, 1, 0));
        ax.Add(new Line2D(new[] { 0.0, 1 }, new[] { 0.0, 1 }, SKColors.Red));
        ax.XLimSet = (0, 1); ax.YLimSet = (0, 1);
        if (equalAspect) { ax.AspectMode = "equal"; ax.AspectValue = 1; }
        using var bmp = SKBitmap.Decode(fig.RenderPng());
        var box = ax.Box;
        Assert.True(DarkPixelNear(bmp, box.Right, box.Bottom + 2), "x tick at the right edge");
        Assert.True(DarkPixelNear(bmp, box.Left - 2, box.Top), "y tick at the top edge");
    }

    [Fact]
    public void ColormapRegistryIsSafeToUseFromManyThreads()
    {
        // Get("<name>_r") creates the reversed map lazily (a write to the registry) while other threads enumerate Names.
        // Writers and readers are released together so the first-time creations overlap with the enumerations.
        var errors = new System.Collections.Concurrent.ConcurrentBag<Exception>();
        var bases = Colormap.Names.Where(n => !n.EndsWith("_r")).ToArray();
        using var go = new Barrier(16);
        var threads = Enumerable.Range(0, 16).Select(t => new Thread(() =>
        {
            try
            {
                go.SignalAndWait();
                if (t % 2 == 0)
                    foreach (var n in bases) Colormap.Get(n + "_r");
                else
                    for (int r = 0; r < 4000; r++) Assert.NotEmpty(Colormap.Names.ToList());
            }
            catch (Exception e) { errors.Add(e); }
        })).ToList();
        threads.ForEach(x => x.Start());
        threads.ForEach(x => x.Join());
        Assert.True(errors.IsEmpty, errors.FirstOrDefault()?.ToString());
    }
}
