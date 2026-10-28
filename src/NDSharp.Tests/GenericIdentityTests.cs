using System.Reflection;
using NDSharp;
using Xunit;
public class GenericIdentityTests
{
    private static T Identity<T>(object value) => (T)typeof(NDArray).Assembly
        .GetType("NDSharp.Red")!.GetMethod("IdentityOf", BindingFlags.NonPublic | BindingFlags.Static)!
        .MakeGenericMethod(typeof(T)).Invoke(null, [value])!;
    [Fact] public void Int64IdentityRetainsLowBits() => Assert.Equal(9007199254740993L, Identity<long>(9007199254740993L));
    [Fact] public void UInt64IdentityRetainsMaximum() => Assert.Equal(ulong.MaxValue, Identity<ulong>(ulong.MaxValue));
    [Fact] public void DecimalIdentityDoesNotUseDouble() => Assert.Equal(1.0000000000000000000000001m, Identity<decimal>(1.0000000000000000000000001m));
    [Fact] public void BooleanIdentityUsesGenericConstants() { Assert.Equal(1, Identity<int>(true)); Assert.Equal(0, Identity<int>(false)); }
}