using NDSharp;
using Xunit;
public class TypedViewTests
{
    [Fact] public void ReverseSliceSharesStorage()
    {
        var x=new NDArray<long>(new long[]{1,2,3,4,5},5);
        var reverse=x.Slice(0,4,3,-2);
        Assert.Equal(new long[]{5,3,1},reverse.ToArray());
        reverse[1]=99; Assert.Equal(99L,x[2]); Assert.Equal(3,reverse.Length);
    }
    [Fact] public void SliceTransposeComposition()
    {
        var x=new NDArray<int>(new int[]{1,2,3,4,5,6},2,3);
        Assert.Equal(new int[]{5,6,2,3},x.Slice(1,1,2).Transpose().Slice(1,1,2,-1).Transpose().ToArray());
    }
    [Fact] public void RowBroadcast()
    {
        var x=new NDArray<int>(new int[]{1,2,3,4,5,6},2,3);
        var row=new NDArray<int>(new int[]{10,20,30},3);
        var result=x.Add(row);
        Assert.Equal(new int[]{11,22,33,14,25,36},result.ToArray());
        result[0,0]=99; Assert.Equal(1,x[0,0]);
    }
    [Fact] public void ColumnAndScalarBroadcast()
    {
        var row=new NDArray<decimal>(new decimal[]{1,2,3},1,3);
        var column=new NDArray<decimal>(new decimal[]{10,20},2,1);
        Assert.Equal(new decimal[]{11,12,13,21,22,23},row.Add(column).ToArray());
        Assert.Equal(new decimal[]{3,4,5},row.Add(new NDArray<decimal>(new decimal[]{2})).ToArray());
    }
    [Fact] public void BroadcastReversedView()
    {
        var x=new NDArray<int>(new int[]{1,2,3},3).Slice(0,2,3,-1);
        Assert.Equal(new int[]{13,12,11},x.Add(new NDArray<int>(new int[]{10})).ToArray());
    }
    [Fact] public void EmptyBroadcastAndSlice()
    {
        var x=new NDArray<int>(Array.Empty<int>(),0,3);
        Assert.Equal(new int[]{0,3},x.Add(new NDArray<int>(new int[]{1,2,3},1,3)).Shape);
        Assert.Empty(new NDArray<int>(new int[]{1,2},2).Slice(0,2,0).ToArray());
    }
    [Fact] public void InvalidArguments()
    {
        var x=new NDArray<int>(new int[]{1,2,3},3);
        Assert.Throws<ArgumentOutOfRangeException>(()=>x.Slice(0,0,2,-1));
        Assert.Throws<ArgumentOutOfRangeException>(()=>x.Slice(0,0,1,0));
        Assert.Throws<ArgumentException>(()=>x.Add(new NDArray<int>(new int[]{1,2},2)));
    }
}