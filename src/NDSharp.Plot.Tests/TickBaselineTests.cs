using NDSharp.Plot;
using Xunit;
namespace NDSharp.Plot.Tests;
public class TickBaselineTests
{
    [Fact]
    public void TextMetricsHaveAscentAndDescentForBaselineAlignment()
    {
        var size=TextMetrics.Measure("0",10,100);
        Assert.True(size.Ascent>0);
        Assert.True(size.Descent>0);
        var upper=size.Ascent/2;
        var lower=size.Ascent/2+size.Descent;
        Assert.Equal(size.Height,upper+lower,10);
        Assert.True(lower>upper);
    }
}