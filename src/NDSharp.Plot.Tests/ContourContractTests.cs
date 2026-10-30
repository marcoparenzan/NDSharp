using SkiaSharp;

namespace NDSharp.Plot.Tests;

/// <summary>
/// Pins the observable input contract of <see cref="ContourArtist"/> (R05). Lines and filled contours deliberately differ:
/// lines tolerate what the shared tracer tolerates, filled goes through the sampler and band lookup, which validate.
/// Where matplotlib behaves differently the difference is stated in the comment.
/// </summary>
public class ContourContractTests
{
    static readonly double[] Ramp = [0, 1, 2, 0, 1, 2, 0, 1, 2];
    static readonly double[] Axis = [0, 1, 2];

    static byte[] Render(ContourArtist? contour, double maxX = 2, double maxY = 2)
    {
        var figure = new Figure();
        var axes = figure.AddAxes(GridCell.Single(1, 1, 0));
        axes.XLimSet = (0, maxX); axes.YLimSet = (0, maxY);
        if (contour is not null) axes.Add(contour);
        return figure.RenderPng();
    }

    static void AssertPaintsNothing(ContourArtist contour) => Assert.Equal(Render(null), Render(contour));

    static ContourArtist Make(double[]? xs = null, double[]? z = null, double[]? levels = null, bool filled = false) =>
        new() { Xs = xs ?? Axis, Ys = Axis, Z = z ?? Ramp, Levels = levels ?? [0.5], Filled = filled, Colors = [SKColors.Red, SKColors.Blue] };

    [Fact]
    public void AllMissingFieldPaintsNothingInBothModes()
    {
        var nan = Enumerable.Repeat(double.NaN, 9).ToArray();
        AssertPaintsNothing(Make(z: nan));
        AssertPaintsNothing(Make(z: nan, levels: [0, 1], filled: true));
    }

    // Axis ordering: lines accept descending or repeated coordinates (matplotlib requires strictly monotonic); filled requires strictly increasing.
    [Fact]
    public void LinesAcceptDescendingAndRepeatedAxes()
    {
        Render(Make(xs: [2, 1, 0]));
        Render(Make(xs: [0, 1, 1]));
    }

    [Theory]
    [InlineData(2.0, 1.0, 0.0)]
    [InlineData(0.0, 1.0, 1.0)]
    public void FilledRejectsAxesThatAreNotStrictlyIncreasing(double a, double b, double c)
    {
        Assert.Throws<ArgumentException>(() => Render(Make(xs: [a, b, c], levels: [0, 1, 2], filled: true)));
    }

    [Fact]
    public void NonFiniteCoordinatesAreRejectedByLines()
    {
        Assert.Throws<ArgumentException>(() => Render(Make(xs: [0, double.NaN, 2])));
    }

    [Fact]
    public void MismatchedFieldLengthIsRejected()
    {
        Assert.Throws<ArgumentException>(() => Render(Make(z: [0, 1, 2])));
    }

    // Level order and values. matplotlib rejects non-increasing levels in both modes.
    [Fact]
    public void LinesAcceptDecreasingAndEmptyLevelsButFilledRejectsThem()
    {
        Render(Make(levels: [1.5, 0.5]));
        Render(Make(levels: []));
        Assert.Throws<ArgumentException>(() => Render(Make(levels: [2, 1, 0], filled: true)));
        Assert.Throws<ArgumentException>(() => Render(Make(levels: [], filled: true)));
    }

    [Fact]
    public void NonFiniteLevelsDifferByMode()
    {
        Render(Make(levels: [double.PositiveInfinity]));                                          // lines: outside the data range, skipped
        Assert.Throws<ArgumentOutOfRangeException>(() => Render(Make(levels: [double.NaN])));      // lines: NaN reaches the tracer
        Assert.Throws<ArgumentException>(() => Render(Make(levels: [0, double.NaN, 2], filled: true)));
    }

    [Fact]
    public void FilledWithASingleLevelHasNoBandAndPaintsNothing()
    {
        AssertPaintsNothing(Make(levels: [1], filled: true));
    }

    [Fact]
    public void InfiniteSamplesAreTolerated()
    {
        Render(Make(z: [0, 1, double.PositiveInfinity, 0, 1, 2, 0, 1, 2], levels: [0, 1, 2], filled: true));
    }

    [Fact]
    public void SingleColumnFieldHasNoCellsAndRendersNothing()
    {
        AssertPaintsNothing(new ContourArtist { Xs = [0], Ys = Axis, Z = [0, 1, 2], Levels = [0.5] });
    }

    [Fact]
    public void FewerColorsThanLevelsCycle()
    {
        // two lines, one colour (Colors = [Red]): both are drawn, so the figure differs from the empty one
        var one = Make(levels: [0.5, 1.5]); one.Colors = [SKColors.Red];
        Assert.NotEqual(Render(null), Render(one));
    }
}
