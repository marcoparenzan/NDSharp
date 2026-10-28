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
