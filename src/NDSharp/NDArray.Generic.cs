using System.Numerics;
namespace NDSharp;

/// <summary>A typed, fixed-shape array. Arithmetic and storage preserve T; transpose creates a shared-storage view.</summary>
public sealed class NDArray<T> where T : INumber<T>
{
    private readonly T[] buffer;
    private readonly int baseOffset;
    private readonly int[] shape;
    private readonly int[] strides;
    public int Rank => shape.Length;
    public int Length => shape.Aggregate(1, (size, dim) => checked(size * dim));
    public int[] Shape => (int[])shape.Clone();
    public int[] Strides => (int[])strides.Clone();

    /// <summary>Copies the values. Empty dimensions are supported; rank zero is a single scalar.</summary>
    public NDArray(ReadOnlySpan<T> values, params int[] dimensions)
    {
        ArgumentNullException.ThrowIfNull(dimensions);
        shape = (int[])dimensions.Clone();
        strides = new int[shape.Length];
        int size = 1;
        for (int i = shape.Length - 1; i >= 0; i--)
        {
            if (shape[i] < 0) throw new ArgumentOutOfRangeException(nameof(dimensions));
            strides[i] = size;
            size = checked(size * shape[i]);
        }
        if (values.Length != size) throw new ArgumentException("Values do not match shape.", nameof(values));
        buffer = values.ToArray();
    }
    private NDArray(T[] values, int[] dimensions, int[] steps, int offset = 0)
    { buffer = values; shape = dimensions; strides = steps; baseOffset = offset; }

    public T this[params int[] indices]
    {
        get => buffer[Offset(indices)];
        set => buffer[Offset(indices)] = value;
    }
    private int Offset(int[] indices)
    {
        ArgumentNullException.ThrowIfNull(indices);
        if (indices.Length != Rank) throw new ArgumentException("Index rank mismatch.", nameof(indices));
        int offset = baseOffset;
        for (int i = 0; i < Rank; i++)
        {
            if ((uint)indices[i] >= (uint)shape[i]) throw new ArgumentOutOfRangeException(nameof(indices));
            offset += indices[i] * strides[i];
        }
        return offset;
    }
    /// <summary>Permutes the axes; omitted axes reverse their order. Writes are shared with the source.</summary>
    public NDArray<T> Transpose(params int[] axes)
    {
        ArgumentNullException.ThrowIfNull(axes);
        if (axes.Length == 0) axes = Enumerable.Range(0, Rank).Reverse().ToArray();
        if (axes.Length != Rank || axes.Distinct().Count() != Rank || axes.Any(a => (uint)a >= (uint)Rank))
            throw new ArgumentException("Expected an axis permutation.", nameof(axes));
        return new NDArray<T>(buffer, axes.Select(a => shape[a]).ToArray(), axes.Select(a => strides[a]).ToArray(), baseOffset);
    }
    /// <summary>Creates a shared-storage slice along one axis. Start is explicit, length is a count, step may be negative.</summary>
    public NDArray<T> Slice(int axis, int start, int length, int step = 1)
    {
        if ((uint)axis >= (uint)Rank) throw new ArgumentOutOfRangeException(nameof(axis));
        if (length < 0) throw new ArgumentOutOfRangeException(nameof(length));
        if (step == 0) throw new ArgumentOutOfRangeException(nameof(step));
        if (length == 0)
        {
            if (start < 0 || start > shape[axis]) throw new ArgumentOutOfRangeException(nameof(start));
        }
        else
        {
            long last = start + (long)(length - 1) * step;
            if (start < 0 || start >= shape[axis] || last < 0 || last >= shape[axis])
                throw new ArgumentOutOfRangeException(nameof(start));
        }
        var dimensions = Shape; var steps = Strides;
        dimensions[axis] = length;
        steps[axis] = checked(steps[axis] * step);
        var offset = length == 0 ? baseOffset : checked(baseOffset + start * strides[axis]);
        return new NDArray<T>(buffer, dimensions, steps, offset);
    }

    /// <summary>Copies in logical C order, honoring view strides.</summary>
    public T[] ToArray()
    {
        var result = new T[Length];
        for (int linear = 0; linear < Length; linear++)
        {
            int remainder = linear, offset = baseOffset;
            for (int axis = Rank - 1; axis >= 0; axis--)
            { offset += (remainder % shape[axis]) * strides[axis]; remainder /= shape[axis]; }
            result[linear] = buffer[offset];
        }
        return result;
    }
    /// <summary>Sequential sum in logical order. Uses the overflow and rounding semantics of T.</summary>
    public T Sum()
    {
        var result = T.Zero;
        foreach (var value in ToArray()) result += value;
        return result;
    }
    /// <summary>Non-conjugating dot product of two rank-one arrays in logical order, accumulated in T.</summary>
    public T Dot(NDArray<T> other)
    {
        ArgumentNullException.ThrowIfNull(other);
        if (Rank != 1 || other.Rank != 1) throw new ArgumentException("Dot requires rank-one arrays.", nameof(other));
        if (shape[0] != other.shape[0]) throw new ArgumentException("Vector lengths differ.", nameof(other));
        var result = T.Zero;
        for (int i = 0; i < shape[0]; i++)
            result += buffer[baseOffset + i * strides[0]] * other.buffer[other.baseOffset + i * other.strides[0]];
        return result;
    }

    /// <summary>Dense rank-two matrix product, honoring strides and returning independent storage.
    /// Accumulation uses T's ordinary rounding and overflow semantics; no batched broadcasting.</summary>
    public NDArray<T> MatMul(NDArray<T> other)
    {
        ArgumentNullException.ThrowIfNull(other);
        if (Rank != 2 || other.Rank != 2) throw new ArgumentException("MatMul requires rank-two arrays.", nameof(other));
        if (shape[1] != other.shape[0]) throw new ArgumentException("Inner dimensions differ.", nameof(other));
        int rows = shape[0], inner = shape[1], columns = other.shape[1];
        var values = new T[checked(rows * columns)];
        for (int row = 0; row < rows; row++)
            for (int column = 0; column < columns; column++)
            {
                var sum = T.Zero;
                int left = baseOffset + row * strides[0];
                int right = other.baseOffset + column * other.strides[1];
                for (int k = 0; k < inner; k++)
                    sum += buffer[left + k * strides[1]] * other.buffer[right + k * other.strides[0]];
                values[row * columns + column] = sum;
            }
        return new NDArray<T>(values.AsSpan(), rows, columns);
    }
    /// <summary>Elementwise addition using trailing-axis broadcasting. Returns independent storage.</summary>
    public NDArray<T> Add(NDArray<T> other)
    {
        ArgumentNullException.ThrowIfNull(other);
        int rank = Math.Max(Rank, other.Rank);
        var dimensions = new int[rank];
        var leftSteps = new int[rank]; var rightSteps = new int[rank];
        for (int axis = 0; axis < rank; axis++)
        {
            int l = axis - (rank - Rank), r = axis - (rank - other.Rank);
            int ld = l < 0 ? 1 : shape[l], rd = r < 0 ? 1 : other.shape[r];
            if (ld != rd && ld != 1 && rd != 1) throw new ArgumentException("Shapes cannot broadcast.", nameof(other));
            dimensions[axis] = ld == 1 ? rd : ld;
            leftSteps[axis] = l < 0 || ld == 1 ? 0 : strides[l];
            rightSteps[axis] = r < 0 || rd == 1 ? 0 : other.strides[r];
        }
        int count = dimensions.Aggregate(1, (n, d) => checked(n * d));
        var values = new T[count];
        for (int linear = 0; linear < count; linear++)
        {
            int remainder = linear, left = baseOffset, right = other.baseOffset;
            for (int axis = rank - 1; axis >= 0; axis--)
            {
                int coordinate = remainder % dimensions[axis]; remainder /= dimensions[axis];
                left += coordinate * leftSteps[axis]; right += coordinate * rightSteps[axis];
            }
            values[linear] = buffer[left] + other.buffer[right];
        }
        return new NDArray<T>(values.AsSpan(), dimensions);
    }
}