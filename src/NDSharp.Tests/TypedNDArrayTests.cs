using NDSharp;
using Xunit;
public class TypedNDArrayTests
{
    [Fact] public void DecimalStorageRetainsSmallIncrement()
    {
        var x = new NDArray<decimal>(new decimal[] { 1m, 0.0000000000000000000000001m }, 2);
        Assert.Equal(1.0000000000000000000000001m, x.Sum());
    }
    [Fact] public void TransposeHonorsStridesAndSharesWrites()
    {
        var x = new NDArray<long>(new long[] { 9007199254740993, 2, 3, 4, 5, 6 }, 2, 3);
        var t = x.Transpose();
        Assert.Equal(new long[] {9007199254740993,4,2,5,3,6}, t.ToArray());
        t[2,1] = 99; Assert.Equal(99L, x[1,2]);
    }
    [Fact] public void AddUsesLogicalOrder()
    {
        var x = new NDArray<int>(new int[] {1,2,3,4},2,2);
        Assert.Equal(new int[] {2,5,5,8},x.Add(x.Transpose()).ToArray());
    }
    [Fact] public void ShapeAndInputAreCopied()
    {
        int[] data=[1,2]; int[] shape=[2];
        var x=new NDArray<int>(data,shape); data[0]=9; shape[0]=9;
        var returned=x.Shape; returned[0]=9;
        Assert.Equal(1,x[0]); Assert.Equal(2,x.Shape[0]);
    }
    [Fact] public void ScalarAndEmptyArrays()
    {
        Assert.Equal(7,new NDArray<int>(new int[]{7}).Sum());
        Assert.Empty(new NDArray<int>(Array.Empty<int>(),0,3).Transpose().ToArray());
    }
    [Fact] public void InvalidShapeAxesAndIndices()
    {
        Assert.Throws<ArgumentException>(()=>new NDArray<int>(new int[]{1},2));
        var x=new NDArray<int>(new int[]{1,2},1,2);
        Assert.Throws<ArgumentException>(()=>x.Transpose(0,0));
        Assert.Throws<ArgumentOutOfRangeException>(()=>x[0,2]);
    }
}