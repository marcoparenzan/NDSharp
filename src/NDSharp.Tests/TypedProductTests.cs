using NDSharp;
using Xunit;
public class TypedProductTests
{
    [Fact] public void DotHandlesReverseSlice()
    {
        var x=new NDArray<long>(new long[]{1,2,3},3);
        Assert.Equal(10L,x.Dot(x.Slice(0,2,3,-1)));
    }
    [Fact] public void MatMulRectangular()
    {
        var a=new NDArray<int>(new int[]{1,2,3,4,5,6},2,3);
        var b=new NDArray<int>(new int[]{7,8,9,10,11,12},3,2);
        var c=a.MatMul(b);
        Assert.Equal(new int[]{58,64,139,154},c.ToArray());
        c[0,0]=0; Assert.Equal(1,a[0,0]); Assert.Equal(7,b[0,0]);
    }
    [Fact] public void MatMulViews()
    {
        var a=new NDArray<int>(new int[]{1,2,3,4,5,6},2,3);
        Assert.Equal(new int[]{14,32,32,77},a.MatMul(a.Transpose()).ToArray());
        var reversed=a.Slice(1,2,3,-1);
        Assert.Equal(new int[]{14,32,32,77},reversed.MatMul(reversed.Transpose()).ToArray());
    }
    [Fact] public void EmptyInnerDimensionProducesZero()
    {
        var a=new NDArray<int>(Array.Empty<int>(),2,0);
        var b=new NDArray<int>(Array.Empty<int>(),0,3);
        Assert.Equal(new int[]{0,0,0,0,0,0},a.MatMul(b).ToArray());
        Assert.Equal(0,new NDArray<int>(Array.Empty<int>(),0).Dot(new NDArray<int>(Array.Empty<int>(),0)));
    }
    [Fact] public void RankAndDimensionErrors()
    {
        var a=new NDArray<int>(new int[]{1,2},2);
        var b=new NDArray<int>(new int[]{1},1);
        Assert.Throws<ArgumentException>(()=>a.Dot(b));
        Assert.Throws<ArgumentException>(()=>a.MatMul(b));
        Assert.Throws<ArgumentException>(()=>new NDArray<int>(new int[]{1,2},1,2).MatMul(new NDArray<int>(new int[]{1},1,1)));
    }
}