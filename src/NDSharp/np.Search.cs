// Copyright (c) 2026 Marco Parenzan
//
// Licensed under the MIT License. See the LICENSE file in the project
// root for full license information.

namespace NDSharp;

public static partial class np
{
    /// <summary>numpy <c>searchsorted</c>: for each element of <paramref name="v"/>, the index into the sorted 1-D array <paramref name="a"/> where it would be inserted to keep the order
    /// (<paramref name="side"/> <c>"left"</c>: before equal elements; <c>"right"</c>: after them). NaN sorts after every number. With <paramref name="sorter"/> (indices that sort
    /// <paramref name="a"/>, as <c>argsort</c> gives them) the array need not be sorted; the result then indexes the sorted order. The two arrays are searched in their common dtype.
    /// Real dtypes only (bool counts as uint8); complex input is not implemented. Runs on the MathemaSharp kernels (binary search).</summary>
    public static NDArray SearchSorted(NDArray a, NDArray v, string side = "left", NDArray? sorter = null)
    {
        if (side is not ("left" or "right")) throw new NDValueException($"side must be 'left' or 'right' (got '{side}')");
        if (a.Ndim != 1) throw new NDValueException("object too deep for desired array");
        if (sorter is not null)
        {
            if (sorter.Ndim != 1 || sorter.Size != a.Size) throw new NDValueException("sorter.size must equal a.size");
            a = a.Get(sorter.AsType(DType.Int64));
        }
        var rt = DTypes.Promote(a.DType, v.DType);
        if (rt == DType.Bool) rt = DType.UInt8;
        if (rt.IsComplex() || rt.IsTemporal()) throw new NDNotSupportedException($"searchsorted is not implemented for {rt}");
        var found = KernelPath.SearchSorted(a.CastTo(rt), v.CastTo(rt), side == "right");
        return found ?? throw new NDNotSupportedException($"searchsorted is not implemented for {rt}");
    }
}
