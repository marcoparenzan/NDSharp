using System.Numerics;
using MathemaSharp.LinearAlgebra;
using NDSharp;

namespace NDSharp.Mathema;

/// <summary>Adapters from typed ndarray views to existing dense kernels; copies preserve T and logical order.</summary>
public static class TypedLinearAlgebraBridge
{
    /// <summary>Copies a rank-two view into a rectangular CLR array.</summary>
    public static T[,] ToMatrix<T>(this NDArray<T> array) where T : INumber<T>
    {
        ArgumentNullException.ThrowIfNull(array);
        if (array.Rank != 2) throw new ArgumentException("Expected rank two.", nameof(array));
        var shape = array.Shape;
        var values = array.ToArray();
        var matrix = new T[shape[0], shape[1]];
        for (int i = 0; i < shape[0]; i++)
            for (int j = 0; j < shape[1]; j++) matrix[i, j] = values[i * shape[1] + j];
        return matrix;
    }

    /// <summary>Copies a rectangular CLR matrix to typed storage.</summary>
    public static NDArray<T> ToTypedNDArray<T>(this T[,] matrix) where T : INumber<T>
    {
        ArgumentNullException.ThrowIfNull(matrix);
        var values = new T[matrix.Length];
        int columns = matrix.GetLength(1);
        for (int i = 0; i < matrix.GetLength(0); i++)
            for (int j = 0; j < columns; j++) values[i * columns + j] = matrix[i, j];
        return new NDArray<T>(values.AsSpan(), matrix.GetLength(0), columns);
    }

    /// <summary>Creates the existing reusable LU factorization from a copied view.</summary>
    public static LuFactorization<T> FactorizeLu<T>(this NDArray<T> matrix) where T : IFloatingPointIeee754<T>
        => LuFactorization<T>.Create(matrix.ToMatrix());

    /// <summary>Solves using an existing LU; RHS must be rank one. Returns independent typed storage.</summary>
    public static NDArray<T> Solve<T>(this LuFactorization<T> factor, NDArray<T> rhs) where T : IFloatingPointIeee754<T>
    {
        ArgumentNullException.ThrowIfNull(factor);
        return factor.Solve(rhs.ToVector()).ToTypedNDArray();
    }

    /// <summary>Delegates to the existing QR least-squares kernel; inherits its rank and representability checks.</summary>
    public static NDArray<T> LeastSquares<T>(this NDArray<T> matrix, NDArray<T> rhs) where T : IFloatingPointIeee754<T>
        => MatrixAlgebra.LeastSquares<T>(matrix.ToMatrix(), rhs.ToVector()).ToTypedNDArray();

    /// <summary>Opt-in dense LU diagnostics. The report retains T; it is not an accuracy certificate.</summary>
    public static LinearSolveReport<T> SolveWithDiagnostics<T>(this NDArray<T> matrix, NDArray<T> rhs)
        where T : IFloatingPointIeee754<T>
        => LinearSystemDiagnostics.Solve<T>(matrix.ToMatrix(), rhs.ToVector());
}
