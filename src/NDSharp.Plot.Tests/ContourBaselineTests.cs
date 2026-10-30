using SkiaSharp;

namespace NDSharp.Plot.Tests;

/// <summary>Pre-migration baseline for the active Skia renderer. PNGs are inspected in memory, never saved.</summary>
public class ContourBaselineTests
{
    static (Figure Figure, Axes Axes) FigureWith(ContourArtist contour, double maxX = 2, double maxY = 2)
    {
        var figure = new Figure();
        var axes = figure.AddAxes(GridCell.Single(1, 1, 0));
        axes.XLimSet = (0, maxX); axes.YLimSet = (0, maxY); axes.Add(contour);
        return (figure, axes);
    }

    static bool RedNear(SKBitmap bitmap, double x, double y)
    {
        for (int row = (int)Math.Round(y) - 2; row <= (int)Math.Round(y) + 2; row++)
            for (int col = (int)Math.Round(x) - 2; col <= (int)Math.Round(x) + 2; col++)
                if (col >= 0 && row >= 0 && col < bitmap.Width && row < bitmap.Height
                    && bitmap.GetPixel(col, row) is { Red: > 200, Green: < 70, Blue: < 70 }) return true;
        return false;
    }

    [Fact]
    public void LineContoursInterpolateNonUniformPhysicalXCoordinates()
    {
        var (figure, axes) = FigureWith(new ContourArtist { Xs = [0, 1, 10], Ys = [0, 1, 4],
            Z = [0, 1, 2, 0, 1, 2, 0, 1, 2], Levels = [1.5], Colors = [SKColors.Red], LineWidth = 2 }, 10, 4);
        using var bitmap = SKBitmap.Decode(figure.RenderPng()) ?? throw new Exception("PNG decode failed.");
        Assert.True(RedNear(bitmap, axes.PxX(5.5), axes.PxY(2)));
        Assert.False(RedNear(bitmap, axes.PxX(7.5), axes.PxY(2))); // global index scaling would be wrong
    }

    [Fact]
    public void NonFiniteCellsAreSkippedButFiniteNeighborDraws()
    {
        var (figure, axes) = FigureWith(new ContourArtist { Xs = [0, 1, 2], Ys = [0, 1, 2],
            Z = [double.NaN, 1, 2, 0, 1, 2, 0, 1, 2], Levels = [0.5], Colors = [SKColors.Red], LineWidth = 2 });
        using var bitmap = SKBitmap.Decode(figure.RenderPng()) ?? throw new Exception("PNG decode failed.");
        Assert.False(RedNear(bitmap, axes.PxX(0.5), axes.PxY(0.5)));
        Assert.True(RedNear(bitmap, axes.PxX(0.5), axes.PxY(1.5)));
    }

    [Fact]
    public void LevelsAtTheObservedRangeEndpointsAreNotDrawn()
    {
        var (figure, axes) = FigureWith(new ContourArtist { Xs = [0, 1, 2], Ys = [0, 1, 2],
            Z = [0, 1, 2, 0, 1, 2, 0, 1, 2], Levels = [0, 2], Colors = [SKColors.Red], LineWidth = 2 });
        using var bitmap = SKBitmap.Decode(figure.RenderPng()) ?? throw new Exception("PNG decode failed.");
        Assert.False(RedNear(bitmap, axes.PxX(0), axes.PxY(1)));
        Assert.False(RedNear(bitmap, axes.PxX(2), axes.PxY(1)));
    }

    [Fact]
    public void FilledBandsUseNonUniformPhysicalAxes()
    {
        var (figure, axes) = FigureWith(new ContourArtist { Xs = [0, 1, 10], Ys = [0, 1, 4],
            Z = [0, 1, 2, 0, 1, 2, 0, 1, 2], Levels = [0, 1, 2], Colors = [SKColors.Blue, SKColors.Red], Filled = true }, 10, 4);
        using var bitmap = SKBitmap.Decode(figure.RenderPng()) ?? throw new Exception("PNG decode failed.");
        Assert.True(RedNear(bitmap, axes.PxX(5.5), axes.PxY(2)));
        var blue = bitmap.GetPixel((int)Math.Round(axes.PxX(0.5)), (int)Math.Round(axes.PxY(2)));
        Assert.True(blue.Blue > 200 && blue.Red < 70);
    }

    // Expected values come from matplotlib 3.11.2: MaxNLocator(n + 1, min_n_ticks=1).tick_values(lo, hi), as contour(n) uses.
    [Theory]
    [InlineData(0, 10, 7, new[] { 0, 1.5, 3, 4.5, 6, 7.5, 9, 10.5 })]
    [InlineData(0, 1, 7, new[] { 0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05 })]
    [InlineData(-1.3, 2.7, 7, new[] { -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3 })]
    [InlineData(0, 2, 7, new[] { 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2 })]
    [InlineData(0.001, 0.0093, 7, new[] { 0, 0.0015, 0.003, 0.0045, 0.006, 0.0075, 0.009, 0.0105 })]
    [InlineData(0, 10, 3, new[] { 0, 2.5, 5, 7.5, 10 })]
    [InlineData(-3, 3, 10, new[] { -3, -2.4, -1.8, -1.2, -0.6, 0, 0.6, 1.2, 1.8, 2.4, 3 })]
    public void AutomaticLevelsMatchMatplotlib(double lo, double hi, int n, double[] expected)
    {
        var levels = ContourArtist.AutoLevels(lo, hi, n);
        Assert.Equal(expected.Length, levels.Length);
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], levels[i], 9);
    }

    [Fact]
    public void RepeatedLevelsDrawTheSameLineAsOneLevel()
    {
        var (figure, axes) = FigureWith(new ContourArtist { Xs = [0, 1, 2], Ys = [0, 1, 2],
            Z = [0, 1, 2, 0, 1, 2, 0, 1, 2], Levels = [0.5, 0.5, 0.5], Colors = [SKColors.Red], LineWidth = 2 });
        using var bitmap = SKBitmap.Decode(figure.RenderPng()) ?? throw new Exception("PNG decode failed.");
        Assert.True(RedNear(bitmap, axes.PxX(0.5), axes.PxY(1)));
        Assert.False(RedNear(bitmap, axes.PxX(1.5), axes.PxY(1)));
    }

    [Fact]
    // Known difference: matplotlib raises ValueError("Contour levels must be increasing"); NDSharp accepts repeats.
    // A repeated level adds a zero-width band that still consumes a colour index, so the upper band takes the third colour.
    public void FilledRepeatedLevelsAddAnEmptyBandThatConsumesAColour()
    {
        var (figure, axes) = FigureWith(new ContourArtist { Xs = [0, 1, 2], Ys = [0, 1, 2],
            Z = [0, 1, 2, 0, 1, 2, 0, 1, 2], Levels = [0, 1, 1, 2], Colors = [SKColors.Blue, SKColors.Red, SKColors.Green], Filled = true });
        using var bitmap = SKBitmap.Decode(figure.RenderPng()) ?? throw new Exception("PNG decode failed.");
        var left = bitmap.GetPixel((int)Math.Round(axes.PxX(0.5)), (int)Math.Round(axes.PxY(1)));
        var right = bitmap.GetPixel((int)Math.Round(axes.PxX(1.5)), (int)Math.Round(axes.PxY(1)));
        Assert.True(left.Blue > 200 && left.Red < 70, $"left was {left}");
        Assert.True(right.Green > 100 && right.Red < 70 && right.Blue < 70, $"right was {right}"); // SKColors.Green is (0,128,0)
    }

    [Fact]
    public void FilledAlphaScalesThePixelOpacity()
    {
        var contour = new ContourArtist { Xs = [0, 1, 2], Ys = [0, 1, 2],
            Z = [0, 1, 2, 0, 1, 2, 0, 1, 2], Levels = [0, 1, 2], Colors = [SKColors.Blue, SKColors.Red], Filled = true, Alpha = 0.5 };
        var (figure, axes) = FigureWith(contour);
        using var bitmap = SKBitmap.Decode(figure.RenderPng()) ?? throw new Exception("PNG decode failed.");
        var pixel = bitmap.GetPixel((int)Math.Round(axes.PxX(0.5)), (int)Math.Round(axes.PxY(1)));
        // composited over the white figure background: blue at 50% is (127,127,255)
        Assert.InRange(pixel.Red, 110, 145);
        Assert.InRange(pixel.Blue, 240, 255);
    }

    [Fact]
    public void FilledContoursLeaveNonFiniteSamplesUnpainted()
    {
        var (figure, axes) = FigureWith(new ContourArtist { Xs = [0, 1, 2], Ys = [0, 1, 2],
            Z = [double.NaN, double.NaN, double.NaN, double.NaN, double.NaN, double.NaN, 0, 1, 2], Levels = [0, 1, 2], Colors = [SKColors.Blue, SKColors.Red], Filled = true });
        using var bitmap = SKBitmap.Decode(figure.RenderPng()) ?? throw new Exception("PNG decode failed.");
        var gap = bitmap.GetPixel((int)Math.Round(axes.PxX(1)), (int)Math.Round(axes.PxY(1.5)));
        Assert.True(gap.Red > 240 && gap.Green > 240 && gap.Blue > 240, $"gap was {gap}");
    }

    [Fact]
    public void FilledContoursRetainSeparateBandRendering()
    {
        var (figure, axes) = FigureWith(new ContourArtist { Xs = [0, 1, 2], Ys = [0, 1, 2],
            Z = [0, 1, 2, 0, 1, 2, 0, 1, 2], Levels = [0, 1, 2], Colors = [SKColors.Blue, SKColors.Red], Filled = true });
        using var bitmap = SKBitmap.Decode(figure.RenderPng()) ?? throw new Exception("PNG decode failed.");
        Assert.True(RedNear(bitmap, axes.PxX(1.5), axes.PxY(1)));
        var blue = bitmap.GetPixel((int)Math.Round(axes.PxX(0.5)), (int)Math.Round(axes.PxY(1)));
        Assert.True(blue.Blue > 200 && blue.Red < 70);
    }
}
