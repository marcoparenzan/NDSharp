// Copyright (c) 2026 Marco Parenzan
// Licensed under the MIT License. See the LICENSE file in the project root.

namespace NDSharp;

/// <summary>A typed strided view sharing a CLR buffer, with immutable metadata snapshots.
/// Indices are zero-based; strides and offset are in elements. Negative and zero strides
/// are supported. Values, including non-finite numbers, are neither converted nor copied.
/// Buffer writes are shared and require external synchronization.</summary>
public sealed class ArrayView<T>
{
    private readonly int[] shape;
    private readonly int[] strides;

    /// <summary>The shared backing storage; no ownership transfer occurs.</summary>
    public T[] Buffer { get; }
    /// <summary>Nonnegative axis lengths; an empty shape describes one scalar.</summary>
    public ReadOnlySpan<int> Shape => shape;
    /// <summary>Signed element strides, one per axis.</summary>
    public ReadOnlySpan<int> Strides => strides;
    /// <summary>Absolute position of the logical origin in the buffer.</summary>
    public int Offset { get; }
    /// <summary>Number of axes.</summary>
    public int Rank => shape.Length;
    /// <summary>Logical element count, including repeated broadcast elements.</summary>
    public int Size { get; }

    /// <summary>Creates a view without copying elements. Metadata is copied. Every reachable
    /// element must be in the buffer; empty views accept an offset from zero through buffer length.
    /// Negative dimensions, rank mismatch, invalid bounds and counts above Int32.MaxValue are rejected.</summary>
    public ArrayView(T[] buffer, ReadOnlySpan<int> shape, ReadOnlySpan<int> strides, int offset = 0)
    {
        ArgumentNullException.ThrowIfNull(buffer);
        if (shape.Length != strides.Length) throw new ArgumentException("Shape and stride ranks differ.");
        bool empty = false;
        foreach (var length in shape)
        {
            if (length < 0) throw new ArgumentOutOfRangeException(nameof(shape));
            empty |= length == 0;
        }
        long count = empty ? 0 : 1;
        if (!empty)
            foreach (var length in shape)
            {
                count = checked(count * length);
                if (count > int.MaxValue) throw new ArgumentOutOfRangeException(nameof(shape));
            }
        long minimum = offset, maximum = offset;
        if (!empty)
            for (int axis = 0; axis < shape.Length; axis++)
            {
                long extent = (long)(shape[axis] - 1) * strides[axis];
                minimum = checked(minimum + Math.Min(extent, 0));
                maximum = checked(maximum + Math.Max(extent, 0));
            }
        if (empty ? offset < 0 || offset > buffer.Length : minimum < 0 || maximum >= buffer.Length)
            throw new ArgumentOutOfRangeException(nameof(offset), "View reaches outside the buffer.");
        Buffer = buffer;
        this.shape = shape.ToArray();
        this.strides = strides.ToArray();
        Offset = offset;
        Size = (int)count;
    }

    /// <summary>Returns a shared element by a full zero-based index; negative indices are rejected.
    /// No arithmetic is performed on the value, preserving its bits and special values.</summary>
    public ref T this[params int[] indices]
    {
        get
        {
            ArgumentNullException.ThrowIfNull(indices);
            if (indices.Length != Rank) throw new ArgumentException("Index rank differs.", nameof(indices));
            long position = Offset;
            for (int axis = 0; axis < Rank; axis++)
            {
                if ((uint)indices[axis] >= (uint)shape[axis]) throw new ArgumentOutOfRangeException(nameof(indices));
                position += (long)indices[axis] * strides[axis];
            }
            return ref Buffer[(int)position];
        }
    }
}
