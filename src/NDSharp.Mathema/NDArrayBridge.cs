using System.Numerics;
using NDSharp;

namespace NDSharp.Mathema;

/// <summary>
/// Conversions between NDArray (runtime dtype, NumPy semantics) and the types used by the Numbers libraries (T[], T[,], Complex&lt;T&gt;[]).
/// Towards Numbers it always copies in C order converting the dtype: NDSharp's strided views are not exposed as Span.
/// Towards NDSharp, <c>double[]</c> is wrapped without copying (modifying one modifies the other); every other format goes through double and may lose digits:
/// NDSharp only knows NumPy's dtypes (float32, float64, complex128...), not Decimal32/64/128.
/// </summary>
public static class NDArrayBridge
{
    /// <summary>1-D array → double[] (copy, with a cast from any numeric dtype).</summary>
    public static double[] ToVector(this NDArray array)
    {
        if (array.Ndim != 1) throw new ArgumentException($"Atteso un array 1-D, ricevuto {array.Ndim}-D.");
        return array.ToArray<double>();
    }

    /// <summary>2-D array (rows = time, columns = series for the VARs) → double[,].</summary>
    public static double[,] ToMatrix(this NDArray array)
    {
        if (array.Ndim != 2) throw new ArgumentException($"Atteso un array 2-D, ricevuto {array.Ndim}-D.");
        var flat = array.ToArray<double>();
        int rows = array.Shape[0], cols = array.Shape[1];
        var result = new double[rows, cols];
        for (var i = 0; i < rows; i++)
            for (var j = 0; j < cols; j++) result[i, j] = flat[i * cols + j];
        return result;
    }

    /// <summary>Array 1-D complex128 → Complex&lt;double&gt;[] (copia).</summary>
    public static Complex<double>[] ToComplexVector(this NDArray array)
    {
        if (array.Ndim != 1) throw new ArgumentException($"Atteso un array 1-D, ricevuto {array.Ndim}-D.");
        var source = array.ToArray<Complex>();
        var result = new Complex<double>[source.Length];
        for (var i = 0; i < source.Length; i++) result[i] = new Complex<double>(source[i].Real, source[i].Imaginary);
        return result;
    }

    /// <summary>1-D complex array (complex64 or complex128) → Complex&lt;T&gt;[] for any IEEE Generic Math format (copy). NDSharp stores both complex dtypes with
    /// double parts, so a T narrower than double rounds and a T wider than double (Decimal128 is not IEEE here) only receives double precision.</summary>
    public static Complex<T>[] ToComplexVector<T>(this NDArray array) where T : IFloatingPointIeee754<T>, IMinMaxValue<T>
    {
        ArgumentNullException.ThrowIfNull(array);
        if (array.Ndim != 1) throw new ArgumentException("Atteso un array 1-D.", nameof(array));
        if (!array.DType.IsComplex()) throw new NotSupportedException($"Complex conversion needs a complex dtype, not {array.DType}.");
        var source = array.ToArray<Complex>();
        var result = new Complex<T>[source.Length];
        for (var i = 0; i < source.Length; i++) result[i] = new Complex<T>(T.CreateChecked(source[i].Real), T.CreateChecked(source[i].Imaginary));
        return result;
    }

    /// <summary>Complex&lt;T&gt;[] → complex128 NDArray (copy; the parts go through double).</summary>
    public static NDArray ToNDArray<T>(this Complex<T>[] values) where T : IFloatingPointIeee754<T>, IMinMaxValue<T>
    {
        ArgumentNullException.ThrowIfNull(values);
        var buffer = new Complex[values.Length];
        for (var i = 0; i < values.Length; i++) buffer[i] = new Complex(double.CreateChecked(values[i].Real), double.CreateChecked(values[i].Imaginary));
        return NDArray.FromArray(buffer);
    }

    /// <summary>1-D array → T[] with Generic Math (copy from the native dtype, checked conversion without double intermediates).</summary>
    public static T[] ToVector<T>(this NDArray array) where T : INumber<T>
    {
        ArgumentNullException.ThrowIfNull(array);
        if (array.Ndim != 1) throw new ArgumentException("Atteso un array 1-D.", nameof(array));
        return ReadNative<T>(array);
    }

    /// <summary>2-D array → T[,] with Generic Math (copy from the native dtype, checked conversion without double intermediates).</summary>
    public static T[,] ToMatrix<T>(this NDArray array) where T : INumber<T>
    {
        ArgumentNullException.ThrowIfNull(array);
        if (array.Ndim != 2) throw new ArgumentException("Atteso un array 2-D.", nameof(array));
        var values = ReadNative<T>(array);
        var result = new T[array.Shape[0], array.Shape[1]];
        for (var i = 0; i < result.GetLength(0); i++)
            for (var j = 0; j < result.GetLength(1); j++) result[i, j] = values[i * result.GetLength(1) + j];
        return result;
    }

    private static T[] ReadNative<T>(NDArray array) where T : INumber<T> => array.DType switch
    {
        DType.Int8 => ConvertNative<sbyte, T>(array),
        DType.UInt8 => ConvertNative<byte, T>(array),
        DType.Int16 => ConvertNative<short, T>(array),
        DType.UInt16 => ConvertNative<ushort, T>(array),
        DType.Int32 => ConvertNative<int, T>(array),
        DType.UInt32 => ConvertNative<uint, T>(array),
        DType.Int64 => ConvertNative<long, T>(array),
        DType.UInt64 => ConvertNative<ulong, T>(array),
        DType.Float16 => ConvertNative<Half, T>(array),
        DType.Float32 => ConvertNative<float, T>(array),
        DType.Float64 => ConvertNative<double, T>(array),
        DType.Bool => array.ToArray<bool>().Select(b => b ? T.One : T.Zero).ToArray(),
        _ => throw new NotSupportedException($"Real conversion not available for {array.DType}.")
    };

    private static T[] ConvertNative<TSource, T>(NDArray array)
        where TSource : unmanaged, INumberBase<TSource> where T : INumber<T>
    {
        var source = array.ToArray<TSource>();
        if (typeof(TSource) == typeof(T)) return (T[])(object)source;   // same type: ToArray already copied, no second pass
        var result = new T[source.Length];
        for (var i = 0; i < source.Length; i++) result[i] = T.CreateChecked(source[i]);
        return result;
    }

    /// <summary>double[] → 1-D float64 NDArray, without copying.</summary>
    public static NDArray ToNDArray(this double[] values) => NDArray.FromArray(values);

    /// <summary>Complex&lt;double&gt;[] → NDArray complex128 (copia).</summary>
    public static NDArray ToNDArray(this Complex<double>[] values)
    {
        var buffer = new Complex[values.Length];
        for (var i = 0; i < values.Length; i++) buffer[i] = new Complex(values[i].Real, values[i].Imaginary);
        return NDArray.FromArray(buffer);
    }

    /// <summary>T[] → float64 NDArray for any IEEE Generic Math format (copy, going through double: Decimal128 loses digits).</summary>
    public static NDArray ToNDArray<T>(this T[] values) where T : INumber<T>
    {
        var buffer = new double[values.Length];
        for (var i = 0; i < values.Length; i++) buffer[i] = double.CreateChecked(values[i]);
        return NDArray.FromArray(buffer);
    }

    /// <summary>Copies values into native generic storage without changing their numeric type.</summary>
    public static NDArray<T> ToTypedNDArray<T>(this T[] values) where T : INumber<T>
    {
        ArgumentNullException.ThrowIfNull(values);
        return new NDArray<T>(values.AsSpan(), values.Length);
    }

    /// <summary>Copies a rank-one typed array in logical order, without dtype conversion.</summary>
    public static T[] ToVector<T>(this NDArray<T> array) where T : INumber<T>
    {
        ArgumentNullException.ThrowIfNull(array);
        if (array.Rank != 1) throw new ArgumentException("Atteso un array 1-D.", nameof(array));
        return array.ToArray();
    }

    /// <summary>double[,] → NDArray float64 2-D (copia).</summary>
    public static NDArray ToNDArray(this double[,] values) => NDArray.FromMultiDim(values);
}
