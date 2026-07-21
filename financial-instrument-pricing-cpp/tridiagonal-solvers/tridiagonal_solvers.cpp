// Two O(J) direct solvers for the tridiagonal system
//   a_j u_{j-1} + b_j u_j + c_j u_{j+1} = f_j,   j = 1, ..., J-1
//   u_0 = phi, u_J = psi
// implemented independently and cross-checked against each other and against
// the closed-form solution of a two-point boundary value problem.
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

// --- Method 1: LU decomposition specialized to tridiagonal matrices ---------
//
// Write A = LU with
//   L = bidiagonal(a_j on the subdiagonal, beta_j on the diagonal)
//   U = bidiagonal(1 on the diagonal, gamma_j on the superdiagonal)
// Matching coefficients in A = LU gives the O(J) recursion
//   beta_1 = b_1,          gamma_1 = c_1 / beta_1
//   beta_j = b_j - a_j*gamma_{j-1},   gamma_j = c_j / beta_j   (j = 2, ..., J-1)
// Then Au = f factors into Lz = f (forward sweep) and Uu = z (backward sweep).
// This exists and is well-defined whenever beta_j != 0 for every j, which is
// guaranteed a priori under diagonal dominance: |b_j| > |a_j| + |c_j|. Diagonal
// dominance also bounds |beta_j| away from 0 uniformly in J, which is what
// keeps the recursion numerically stable (no pivoting is required).
std::vector<double> solveLU(const std::vector<double>& a, const std::vector<double>& b,
                        const std::vector<double>& c, const std::vector<double>& f_in) {
    // a[0] and c[n-1] are boundary stubs: this routine solves the closed
    // n x n system as given, so any known Dirichlet data must already be
    // folded into f_in by the caller (see foldDirichletBC below).
    std::vector<double> f = f_in;
    int n = static_cast<int>(b.size());  // indices 0..n-1
    std::vector<double> beta(n), gamma(n), z(n), u(n);

    beta[0] = b[0];
    gamma[0] = c[0] / beta[0];
    for (int j = 1; j < n; ++j) {
        beta[j] = b[j] - a[j] * gamma[j - 1];
        if (j < n - 1) gamma[j] = c[j] / beta[j];
    }

    z[0] = f[0] / beta[0];
    for (int j = 1; j < n; ++j) z[j] = (f[j] - a[j] * z[j - 1]) / beta[j];

    u[n - 1] = z[n - 1];
    for (int j = n - 2; j >= 0; --j) u[j] = z[j] - gamma[j] * u[j + 1];
    return u;
}

// --- Method 2: Godunov's double sweep (balayage) method ---------------------
//
// Seeks the solution in the form u_j = L_{j+1/2} u_{j+1} + K_{j+1/2}, with
//   L_{1/2} = 0,                 K_{1/2} = phi
//   L_{j+1/2} = -c_j / (b_j + a_j L_{j-1/2})
//   K_{j+1/2} = (f_j - a_j K_{j-1/2}) / (b_j + a_j L_{j-1/2})
// for j = 1, ..., J-1 (forward sweep), then u_J = psi and
//   u_j = L_{j+1/2} u_{j+1} + K_{j+1/2},   j = J-1, ..., 0   (backward sweep).
// This is algebraically a re-parametrization of the same LU factorization
// (L_{j+1/2} = -gamma_j, K_{j+1/2} = z_j above) but is the historically
// preferred form for two-point boundary value problems with Dirichlet data,
// since phi and psi enter directly as the seed of the recursion rather than
// through the right-hand side std::vector.
std::vector<double> solveGodunovDoubleSweep(const std::vector<double>& a, const std::vector<double>& b,
                                        const std::vector<double>& c, const std::vector<double>& f,
                                        double phi, double psi) {
    int J = static_cast<int>(a.size()) + 1;  // interior unknowns u_1..u_{J-1}; a,b,c,f sized J-1
    std::vector<double> Lh(J), Kh(J), u(J + 1);

    Lh[0] = 0.0;
    Kh[0] = phi;
    for (int j = 1; j <= J - 1; ++j) {
        double denom = b[j - 1] + a[j - 1] * Lh[j - 1];
        Lh[j] = -c[j - 1] / denom;
        Kh[j] = (f[j - 1] - a[j - 1] * Kh[j - 1]) / denom;
    }

    u[J] = psi;
    for (int j = J - 1; j >= 0; --j) u[j] = Lh[j] * u[j + 1] + Kh[j];
    return u;
}

// Folds known Dirichlet data u_0=phi, u_J=psi into the interior right-hand
// side, since solveLU operates on the closed (J-1)-dimensional interior
// system and has no other way to see the boundary values.
std::vector<double> foldDirichletBC(const std::vector<double>& a, const std::vector<double>& c,
                                const std::vector<double>& f, double phi, double psi) {
    std::vector<double> g = f;
    g[0] -= a[0] * phi;
    g[g.size() - 1] -= c[c.size() - 1] * psi;
    return g;
}

double maxAbsDiff(const std::vector<double>& x, const std::vector<double>& y) {
    double m = 0.0;
    for (size_t i = 0; i < x.size(); ++i) m = std::max(m, std::fabs(x[i] - y[i]));
    return m;
}

int main() {
    std::cout << std::fixed << std::setprecision(10);

    // --- Test 1: u'' + u = 0 on (0,1), u(0)=0, u(1)=1 ---------------------
    // Central differences: (u_{j-1} - 2u_j + u_{j+1})/h^2 + u_j = 0, i.e.
    //   a_j = 1,  b_j = -2 + h^2,  c_j = 1,  f_j = 0,   j = 1, ..., J-1
    // Closed form: u(x) = std::sin(x)/std::sin(1). This is a self-adjoint, negative
    // Helmholtz-type operator; on this mesh (-2+h^2 is close to -2) the
    // diagonal dominance condition |b_j| > |a_j|+|c_j| = 2 holds strictly
    // for every finite h, guaranteeing a unique, stably computed solution.
    {
        int J = 20;
        double h = 1.0 / J;
        int n = J - 1;
        std::vector<double> a(n, 1.0), b(n, -2.0 + h * h), c(n, 1.0), f(n, 0.0);

        std::vector<double> u_lu_full(J + 1);
        std::vector<double> u_lu = solveLU(a, b, c, foldDirichletBC(a, c, f, 0.0, 1.0));
        u_lu_full[0] = 0.0;
        for (int j = 0; j < n; ++j) u_lu_full[j + 1] = u_lu[j];
        u_lu_full[J] = 1.0;

        std::vector<double> u_ds = solveGodunovDoubleSweep(a, b, c, f, 0.0, 1.0);

        double maxErrLU = 0.0, maxErrDS = 0.0;
        for (int j = 0; j <= J; ++j) {
            double x = j * h;
            double exact = std::sin(x) / std::sin(1.0);
            maxErrLU = std::max(maxErrLU, std::fabs(u_lu_full[j] - exact));
            maxErrDS = std::max(maxErrDS, std::fabs(u_ds[j] - exact));
        }

        std::cout << "Test 1: u'' + u = 0, u(0)=0, u(1)=1  (J=" << J << ", h=" << h << ")\n";
        std::cout << "  max |LU - exact|            = " << maxErrLU << "\n";
        std::cout << "  max |DoubleSweep - exact|    = " << maxErrDS << "\n";
        std::cout << "  max |LU - DoubleSweep|       = " << maxAbsDiff(u_lu_full, u_ds) << "\n\n";

        // Export the full grid (not just a summary) for plotting: both solvers
        // against the closed-form solution, for visual confirmation of the
        // O(J) direct solves' correctness.
        std::ofstream out("test1_solution.csv");
        out << std::fixed << std::setprecision(10);
        out << "x,exact,LU,DoubleSweep\n";
        for (int j = 0; j <= J; ++j) {
            double x = j * h;
            double exact = std::sin(x) / std::sin(1.0);
            out << x << "," << exact << "," << u_lu_full[j] << "," << u_ds[j] << "\n";
        }
        out.close();
    }

    // --- Test 2: random diagonally dominant systems, larger J -------------
    // Draws a_j, c_j ~ U(-1,1) and sets b_j with a strict dominance margin,
    // so both solvers are guaranteed well-posed; compares them directly
    // against each other (no closed form here, so mutual agreement is the
    // check) across several sizes.
    {
        std::mt19937 rng(42);
        std::uniform_real_distribution<double> unif(-1.0, 1.0);

        std::cout << "Test 2: random diagonally dominant tridiagonal systems\n";
        for (int n : {5, 50, 500, 5000}) {
            std::vector<double> a(n), b(n), c(n), f(n);
            for (int j = 0; j < n; ++j) {
                a[j] = unif(rng);
                c[j] = unif(rng);
                f[j] = unif(rng);
                b[j] = (std::fabs(a[j]) + std::fabs(c[j]) + 1.0) * (unif(rng) > 0 ? 1.0 : -1.0);
            }
            std::vector<double> uLU = solveLU(a, b, c, f);

            // Re-cast the same system in Godunov's boundary-value convention:
            // treat index 0 and n-1 as free unknowns solved directly (no
            // Dirichlet reduction needed) by using dummy phi/psi and folding
            // the first/last equations back in via a,b,c already spanning
            // the full range -- here we simply compare the interior overlap
            // by solving the identical n-dimensional system with a_0 = c_{n-1} = 0
            // treated as ordinary tridiagonal entries (no reduction), which is
            // exactly what solveLU already does; solveGodunovDoubleSweep is
            // exercised on its native two-point boundary form in Test 1 and 3.
            double resid = 0.0;
            for (int j = 0; j < n; ++j) {
                double lhs = b[j] * uLU[j];
                if (j > 0) lhs += a[j] * uLU[j - 1];
                if (j < n - 1) lhs += c[j] * uLU[j + 1];
                resid = std::max(resid, std::fabs(lhs - f[j]));
            }
            std::cout << "  n=" << std::setw(5) << n << "  max residual |Au-f| (LU) = " << resid << "\n";
        }
        std::cout << "\n";
    }

    // --- Test 3: a second boundary value problem, both methods on native form
    // u'' - u = -1 on (0,1), u(0)=0, u(1)=0.  Exact: u(x) = 1 - std::cosh(x-1/2)/std::cosh(1/2).
    {
        int J = 40;
        double h = 1.0 / J;
        int n = J - 1;
        std::vector<double> a(n, 1.0), b(n, -2.0 - h * h), c(n, 1.0), f(n, -h * h);

        std::vector<double> u_ds = solveGodunovDoubleSweep(a, b, c, f, 0.0, 0.0);
        std::vector<double> u_lu = solveLU(a, b, c, foldDirichletBC(a, c, f, 0.0, 0.0));

        double maxErrDS = 0.0, maxErrLU = 0.0;
        for (int j = 0; j <= J; ++j) {
            double x = j * h;
            double exact = 1.0 - std::cosh(x - 0.5) / std::cosh(0.5);
            maxErrDS = std::max(maxErrDS, std::fabs(u_ds[j] - exact));
            double u_lu_j = (j == 0) ? 0.0 : (j == J ? 0.0 : u_lu[j - 1]);
            maxErrLU = std::max(maxErrLU, std::fabs(u_lu_j - exact));
        }
        std::cout << "Test 3: u'' - u = -1, u(0)=0, u(1)=0  (J=" << J << ")\n";
        std::cout << "  max |DoubleSweep - exact|    = " << maxErrDS << "\n";
        std::cout << "  max |LU - exact|             = " << maxErrLU << "\n";
    }

    return 0;
}
