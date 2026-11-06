// Copyright (c) 2026 Marco Parenzan
//
// Licensed under the MIT License. See the LICENSE file in the project
// root for full license information.

namespace NDSharp;

public static partial class np
{
    // ================================================================ partition, argpartition, lexsort (on the sorting kernels)

    private static void CheckKth(int[] kth, int n)
    {
        foreach (var k in kth)
            if (k < -n || k >= n) throw new NDValueException($"kth(={k}) out of bounds ({n})");
    }

    /// <summary>numpy <c>partition</c>: a copy of <paramref name="a"/> in which, along <paramref name="axis"/> (<c>null</c>: the flattened array), the element at every position in
    /// <paramref name="kth"/> is the one a full sort would put there, with no larger element before it and no smaller one after it (NaN counts as the largest). The order of the
    /// other elements is unspecified in numpy; here the whole lane comes out sorted, which satisfies the same guarantee and is deterministic (numpy&apos;s introselect leaves a different,
    /// implementation-specific order). Runs on the MathemaSharp sorting kernels.</summary>
    public static NDArray Partition(NDArray a, int[] kth, int? axis = -1)
    {
        if (axis is null) return Partition(Flatten(a), kth, 0);
        if (a.Ndim == 0) throw new NDAxisException(axis.Value, 0);
        int ax = NormalizeAxis(axis.Value, a.Ndim);
        CheckKth(kth, a.Shape[ax]);
        return Sort(a, ax);
    }

    /// <summary>numpy <c>partition</c> with a single <paramref name="kth"/>.</summary>
    public static NDArray Partition(NDArray a, int kth, int? axis = -1) => Partition(a, new[] { kth }, axis);

    /// <summary>numpy <c>argpartition</c>: the indices that would partition <paramref name="a"/> as <see cref="Partition(NDArray, int[], int?)"/> does. Here they are the stable
    /// <c>argsort</c> indices (see the remark there on the order of the other elements).</summary>
    public static NDArray ArgPartition(NDArray a, int[] kth, int? axis = -1)
    {
        if (axis is null) return ArgPartition(Flatten(a), kth, 0);
        if (a.Ndim == 0) throw new NDAxisException(axis.Value, 0);
        int ax = NormalizeAxis(axis.Value, a.Ndim);
        CheckKth(kth, a.Shape[ax]);
        return ArgSort(a, ax);
    }

    /// <summary>numpy <c>argpartition</c> with a single <paramref name="kth"/>.</summary>
    public static NDArray ArgPartition(NDArray a, int kth, int? axis = -1) => ArgPartition(a, new[] { kth }, axis);

    /// <summary>numpy <c>lexsort</c>: the indices that sort along <paramref name="axis"/> by the last key, ties broken by the one before it, and so on (stable). <paramref name="keys"/> holds
    /// one key per row of its first axis, all of the same shape. Each pass is a stable argsort on the sorting kernels, applied from the first key to the last.</summary>
    public static NDArray LexSort(NDArray keys, int axis = -1)
    {
        if (keys.Ndim == 0) throw new NDValueException("need sequence of keys with len > 0 in lexsort");
        var list = new List<NDArray>();
        for (int i = 0; i < keys.Shape[0]; i++) list.Add(keys.Get(i).Copy());
        return LexSort(list, axis);
    }

    /// <summary>numpy <c>lexsort</c> over a list of keys of the same shape (see <see cref="LexSort(NDArray, int)"/>).</summary>
    public static NDArray LexSort(IReadOnlyList<NDArray> keys, int axis = -1)
    {
        if (keys.Count == 0) throw new NDValueException("need sequence of keys with len > 0 in lexsort");
        var shape = keys[0].Shape;
        foreach (var k in keys) if (!k.Shape.SequenceEqual(shape)) throw new NDValueException("all keys need to be the same shape");
        if (shape.Length == 0) throw new NDAxisException(axis, 0);
        int ax = NormalizeAxis(axis, shape.Length);
        int n = shape[ax];
        var moved = keys.Select(k => MoveAxis(k, ax, -1).Copy()).ToList();
        var movedShape = (int[])moved[0].Shape.Clone();
        int size = moved[0].Size;
        var result = new long[size];
        if (size > 0)
        {
            int lanes = size / n;
            var flat = moved.Select(m => Reshape(m, lanes, n)).ToList();
            for (int l = 0; l < lanes; l++)
            {
                var idx = Enumerable.Range(0, n).Select(i => (long)i).ToArray();
                foreach (var key in flat)                                                   // least significant key first
                {
                    var values = key.Get(l).Copy().Get(NDArray.FromArray(idx, n)).Copy();
                    var order = ArgSort(values, 0).Copy().Data<long>();
                    var next = new long[n];
                    for (int i = 0; i < n; i++) next[i] = idx[order[i]];
                    idx = next;
                }
                Array.Copy(idx, 0, result, l * n, n);
            }
        }
        return MoveAxis(new NDArray(DType.Int64, result, movedShape), -1, ax).Copy();
    }
}
