// Copyright (c) 2026 Marco Parenzan
// Licensed under the MIT License. See the LICENSE file in the project root.

namespace NDSharp;

public sealed partial class NDArray
{
    /// <summary>Creates a typed zero-copy view with snapshot metadata. T must exactly match the
    /// CLR storage element type, including signedness. No conversion or special-value handling occurs.</summary>
    public ArrayView<T> AsArrayView<T>()
    {
        if (Buffer.GetType() != typeof(T[]))
            throw new NDTypeException($"array of dtype {DType.Name()} is not backed by {typeof(T).Name}[]");
        return new((T[])Buffer, Shape, Strides, Offset);
    }

    /// <summary>Creates an NDArray sharing a typed view's storage with copied metadata.
    /// The explicit dtype must match T exactly; this preserves temporal and complex dtype distinctions.
    /// No values are converted. The resulting array keeps the buffer alive independently of the view.</summary>
    public static NDArray FromArrayView<T>(ArrayView<T> view, DType dtype)
    {
        ArgumentNullException.ThrowIfNull(view);
        if (dtype.ClrType() != typeof(T)) throw new NDTypeException("dtype and view element type differ");
        return new(dtype, view.Buffer, view.Shape.ToArray(), view.Strides.ToArray(), view.Offset, null);
    }
}
