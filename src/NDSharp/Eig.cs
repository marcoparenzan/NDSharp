using System.Numerics;

namespace NDSharp;

/// <summary>
/// General (non-symmetric) eigenvalue problem for real or complex square matrices, numpy's <c>linalg.eig</c> / <c>eigvals</c>.
/// <para>Method: diagonal scaling (balancing), reduction to upper Hessenberg form by Householder reflectors, then the shifted QR
/// algorithm in complex arithmetic (Wilkinson shift, implicit single-shift bulge chase, deflation) to the complex Schur form
/// A = Z T Z^H. Eigenvalues are the diagonal of T; eigenvectors come from back-substitution on the triangular T, multiplied by Z.
/// Working in complex arithmetic handles complex-conjugate pairs of real matrices without the double-shift bookkeeping of the
/// real algorithm; real input is cleaned up afterwards so that real eigenvalues come back exactly real.</para>
/// </summary>
internal static class EigSolver
{
    const double Eps = 2.220446049250313e-16;

    /// <summary>Eigen-decomposition of one n×n matrix given row-major. <paramref name="vectors"/> false skips the eigenvectors.</summary>
    public static (Complex[] w, Complex[]? v) Solve(Complex[] a0, int n, bool vectors, bool realInput)
    {
        var w = new Complex[n];
        if (n == 0) return (w, vectors ? Array.Empty<Complex>() : null);
        if (n == 1)
        {
            w[0] = a0[0];
            return (w, vectors ? new[] { Complex.One } : null);
        }

        double anorm = 0;
        foreach (var x in a0) anorm += x.Real * x.Real + x.Imaginary * x.Imaginary;
        anorm = Math.Sqrt(anorm);
        if (anorm == 0)
        {
            // the zero matrix: every eigenvalue is 0 and the unit vectors are eigenvectors
            if (!vectors) return (w, null);
            var id = new Complex[n * n];
            for (int i = 0; i < n; i++) id[i * n + i] = Complex.One;
            return (w, id);
        }

        var h = (Complex[])a0.Clone();
        var d = new double[n];
        Balance(h, n, d);

        Complex[]? z = null;
        if (vectors)
        {
            z = new Complex[n * n];
            for (int i = 0; i < n; i++) z[i * n + i] = Complex.One;
        }
        Hessenberg(h, n, z);
        if (!Schur(h, n, z, anorm))
            throw new NDLinAlgException("Eigenvalues did not converge");

        for (int i = 0; i < n; i++) w[i] = h[i * n + i];

        Complex[]? v = null;
        if (vectors) v = Eigenvectors(h, z!, n, d, anorm);

        if (realInput) CleanReal(w, v, n, anorm);
        return (w, v);
    }

    // ---------------------------------------------------------------- balancing
    // Parlett-Reinsch scaling with powers of two (exact in floating point): A' = D^-1 A D, d = diag(D).
    static void Balance(Complex[] a, int n, double[] d)
    {
        Array.Fill(d, 1.0);
        const double radix = 2, sqrdx = radix * radix;
        bool done = false;
        while (!done)
        {
            done = true;
            for (int i = 0; i < n; i++)
            {
                double c = 0, r = 0;
                for (int j = 0; j < n; j++)
                {
                    if (j == i) continue;
                    c += Complex.Abs(a[j * n + i]);
                    r += Complex.Abs(a[i * n + j]);
                }
                if (c == 0 || r == 0) continue;
                double g = r / radix, f = 1, s = c + r;
                while (c < g) { f *= radix; c *= sqrdx; }
                g = r * radix;
                while (c > g) { f /= radix; c /= sqrdx; }
                if ((c + r) / f < 0.95 * s)
                {
                    done = false;
                    double gi = 1 / f;
                    d[i] *= f;
                    for (int j = 0; j < n; j++) a[i * n + j] *= gi;     // row i / f
                    for (int j = 0; j < n; j++) a[j * n + i] *= f;      // column i * f
                }
            }
        }
    }

    // ---------------------------------------------------------------- Hessenberg reduction
    static void Hessenberg(Complex[] a, int n, Complex[]? z)
    {
        var v = new Complex[n];
        for (int k = 0; k < n - 2; k++)
        {
            int m = n - k - 1;                                   // length of the column below the diagonal
            double alpha = 0;
            for (int i = 0; i < m; i++) alpha += a[(k + 1 + i) * n + k].Real * a[(k + 1 + i) * n + k].Real + a[(k + 1 + i) * n + k].Imaginary * a[(k + 1 + i) * n + k].Imaginary;
            alpha = Math.Sqrt(alpha);
            // only the entries below the first subdiagonal need annihilating
            double below = 0;
            for (int i = 1; i < m; i++) below += a[(k + 1 + i) * n + k].Real * a[(k + 1 + i) * n + k].Real + a[(k + 1 + i) * n + k].Imaginary * a[(k + 1 + i) * n + k].Imaginary;
            if (below == 0) continue;

            Complex x0 = a[(k + 1) * n + k];
            Complex phase = x0 == Complex.Zero ? Complex.One : x0 / Complex.Abs(x0);
            for (int i = 0; i < m; i++) v[i] = a[(k + 1 + i) * n + k];
            v[0] += phase * alpha;
            double vv = 0;
            for (int i = 0; i < m; i++) vv += v[i].Real * v[i].Real + v[i].Imaginary * v[i].Imaginary;
            if (vv == 0) continue;
            double beta = 2 / vv;

            // left: A[k+1.., k..] -= beta v (v^H A[k+1.., k..])
            for (int j = k; j < n; j++)
            {
                Complex s = Complex.Zero;
                for (int i = 0; i < m; i++) s += Complex.Conjugate(v[i]) * a[(k + 1 + i) * n + j];
                s *= beta;
                for (int i = 0; i < m; i++) a[(k + 1 + i) * n + j] -= v[i] * s;
            }
            // right: A[.., k+1..] -= beta (A[.., k+1..] v) v^H
            for (int i = 0; i < n; i++)
            {
                Complex s = Complex.Zero;
                for (int j = 0; j < m; j++) s += a[i * n + k + 1 + j] * v[j];
                s *= beta;
                for (int j = 0; j < m; j++) a[i * n + k + 1 + j] -= s * Complex.Conjugate(v[j]);
            }
            if (z != null)
                for (int i = 0; i < n; i++)
                {
                    Complex s = Complex.Zero;
                    for (int j = 0; j < m; j++) s += z[i * n + k + 1 + j] * v[j];
                    s *= beta;
                    for (int j = 0; j < m; j++) z[i * n + k + 1 + j] -= s * Complex.Conjugate(v[j]);
                }
            for (int i = k + 2; i < n; i++) a[i * n + k] = Complex.Zero;
        }
    }

    // ---------------------------------------------------------------- shifted QR iteration
    static bool Schur(Complex[] h, int n, Complex[]? z, double anorm)
    {
        int hi = n - 1, iter = 0, total = 0;
        int maxTotal = 60 * n + 120;
        while (hi >= 0)
        {
            // look for a negligible subdiagonal entry to split the problem
            int l = hi;
            while (l > 0)
            {
                double s = Complex.Abs(h[(l - 1) * n + l - 1]) + Complex.Abs(h[l * n + l]);
                if (s == 0) s = anorm;
                if (Complex.Abs(h[l * n + l - 1]) <= Eps * s) { h[l * n + l - 1] = Complex.Zero; break; }
                l--;
            }
            if (l == hi) { hi--; iter = 0; continue; }          // an eigenvalue has converged
            if (++total > maxTotal) return false;
            iter++;

            Complex mu;
            if (iter % 10 == 0)
            {
                // exceptional shift: breaks cycles on pathological matrices
                double ex = Complex.Abs(h[hi * n + hi - 1]) + (hi > 1 ? Complex.Abs(h[(hi - 1) * n + hi - 2]) : 0);
                mu = h[hi * n + hi] + ex * 0.75;
            }
            else
            {
                // Wilkinson shift: the eigenvalue of the trailing 2x2 block closer to its last diagonal entry
                Complex p = h[(hi - 1) * n + hi - 1], q = h[(hi - 1) * n + hi], r = h[hi * n + hi - 1], s = h[hi * n + hi];
                Complex half = (p - s) / 2;
                Complex disc = Complex.Sqrt(half * half + q * r);
                Complex mu1 = (p + s) / 2 + disc, mu2 = (p + s) / 2 - disc;
                mu = Complex.Abs(mu1 - s) < Complex.Abs(mu2 - s) ? mu1 : mu2;
            }

            Complex x = h[l * n + l] - mu, y = h[(l + 1) * n + l];
            for (int k = l; k < hi; k++)
            {
                if (k > l) { x = h[k * n + k - 1]; y = h[(k + 1) * n + k - 1]; }
                Givens(x, y, out double c, out Complex sn);

                int j0 = k > l ? k - 1 : k;
                for (int j = j0; j < n; j++)
                {
                    Complex t1 = h[k * n + j], t2 = h[(k + 1) * n + j];
                    h[k * n + j] = c * t1 + sn * t2;
                    h[(k + 1) * n + j] = -Complex.Conjugate(sn) * t1 + c * t2;
                }
                if (k > l) h[(k + 1) * n + k - 1] = Complex.Zero;

                int iEnd = Math.Min(k + 2, hi);
                for (int i = 0; i <= iEnd; i++)
                {
                    Complex t1 = h[i * n + k], t2 = h[i * n + k + 1];
                    h[i * n + k] = c * t1 + Complex.Conjugate(sn) * t2;
                    h[i * n + k + 1] = -sn * t1 + c * t2;
                }
                if (z != null)
                    for (int i = 0; i < n; i++)
                    {
                        Complex t1 = z[i * n + k], t2 = z[i * n + k + 1];
                        z[i * n + k] = c * t1 + Complex.Conjugate(sn) * t2;
                        z[i * n + k + 1] = -sn * t1 + c * t2;
                    }
            }
        }
        // below the diagonal is zero up to rounding: make it exact so T is a clean triangular matrix
        for (int i = 1; i < n; i++) for (int j = 0; j < i; j++) h[i * n + j] = Complex.Zero;
        return true;
    }

    /// <summary>Complex Givens rotation G = [[c, s], [-conj(s), c]] with c real ≥ 0 and G·[x; y] = [r; 0].</summary>
    static void Givens(Complex x, Complex y, out double c, out Complex s)
    {
        double ay = Complex.Abs(y);
        if (ay == 0) { c = 1; s = Complex.Zero; return; }
        double ax = Complex.Abs(x);
        if (ax == 0) { c = 0; s = Complex.Conjugate(y) / ay; return; }
        double r = Math.Sqrt(ax * ax + ay * ay);
        c = ax / r;
        s = x / ax * Complex.Conjugate(y) / r;
    }

    // ---------------------------------------------------------------- eigenvectors
    static Complex[] Eigenvectors(Complex[] t, Complex[] z, int n, double[] d, double anorm)
    {
        var v = new Complex[n * n];                              // row-major: column k is the k-th eigenvector
        var x = new Complex[n];
        double small = Math.Max(Eps * anorm, double.Epsilon);
        for (int k = n - 1; k >= 0; k--)
        {
            Array.Clear(x);
            x[k] = Complex.One;
            Complex lam = t[k * n + k];
            for (int j = k - 1; j >= 0; j--)
            {
                Complex sum = Complex.Zero;
                for (int l = j + 1; l <= k; l++) sum += t[j * n + l] * x[l];
                Complex denom = t[j * n + j] - lam;
                if (Complex.Abs(denom) < small) denom = small;   // repeated eigenvalue: perturb instead of dividing by zero
                x[j] = -sum / denom;
            }
            // v_k = D Z x
            double nrm = 0;
            var col = new Complex[n];
            for (int i = 0; i < n; i++)
            {
                Complex s = Complex.Zero;
                for (int j = 0; j <= k; j++) s += z[i * n + j] * x[j];
                s *= d[i];
                col[i] = s;
                nrm += s.Real * s.Real + s.Imaginary * s.Imaginary;
            }
            nrm = Math.Sqrt(nrm);
            // unit norm, and the component of largest magnitude made real and positive (LAPACK's convention for complex vectors)
            int big = 0;
            for (int i = 1; i < n; i++) if (Complex.Abs(col[i]) > Complex.Abs(col[big])) big = i;
            Complex phase = Complex.Abs(col[big]) == 0 ? Complex.One : Complex.Conjugate(col[big] / Complex.Abs(col[big]));
            for (int i = 0; i < n; i++) v[i * n + k] = col[i] * phase / nrm;
        }
        return v;
    }

    // ---------------------------------------------------------------- real input: exact real eigenvalues
    static void CleanReal(Complex[] w, Complex[]? v, int n, double anorm)
    {
        double tol = 16 * Eps * anorm;
        for (int k = 0; k < n; k++)
        {
            if (Math.Abs(w[k].Imaginary) > tol) continue;
            w[k] = new Complex(w[k].Real, 0);
            if (v == null) continue;
            // a real eigenvalue of a real matrix has a real eigenvector (up to the phase already fixed above)
            for (int i = 0; i < n; i++) v[i * n + k] = new Complex(v[i * n + k].Real, 0);
            double nrm = 0;
            for (int i = 0; i < n; i++) nrm += v[i * n + k].Real * v[i * n + k].Real;
            nrm = Math.Sqrt(nrm);
            if (nrm > 0) for (int i = 0; i < n; i++) v[i * n + k] /= nrm;
        }
    }
}

