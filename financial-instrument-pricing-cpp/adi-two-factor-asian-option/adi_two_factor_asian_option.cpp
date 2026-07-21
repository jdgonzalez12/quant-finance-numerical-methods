// Fixed-strike arithmetic-average Asian call under Black-Scholes dynamics, priced
// by augmenting the state with the running average and solving the resulting
// two-factor PDE with an ADI (Yanenko/fractional-step) finite-difference scheme.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

// Standard tridiagonal (Thomas) solve: sub[i]*x[i-1] + diag[i]*x[i] + sup[i]*x[i] = rhs[i].
// Takes and returns by reference/out-param to avoid reallocating four vectors
// per call -- this routine runs once per grid line per ADI stage per time
// step ((NS+1)+(NA+1) calls per step), so its own allocation overhead is not
// negligible against its O(n) arithmetic cost.
static void thomasSolveInPlace(std::vector<double>& sub, std::vector<double>& diag,
                                std::vector<double>& sup, std::vector<double>& rhs,
                                std::vector<double>& x, int n) {
    for (int i = 1; i < n; ++i) {
        double m = sub[i] / diag[i - 1];
        diag[i] -= m * sup[i - 1];
        rhs[i] -= m * rhs[i - 1];
    }
    x[n - 1] = rhs[n - 1] / diag[n - 1];
    for (int i = n - 2; i >= 0; --i) x[i] = (rhs[i] - sup[i] * x[i + 1]) / diag[i];
}

// Flat, contiguous row-major storage for the (S, A) value grid, replacing a
// vector<vector<double>>. A nested vector<vector<double>> allocates each row
// as a *separate* heap block, so column-direction access (as in ADI's
// A-sweep stage, which walks fixed i across varying k, and the S-sweep
// stage's column writes into Y) touches scattered, non-contiguous memory --
// exactly whichever direction is not "the" contiguous one. A single flat
// buffer with explicit row-major indexing keeps every access, in either
// sweep direction, within one allocation, and removes NS+1 separate heap
// allocations (and NS+1 pointer indirections per access) in favor of one.
struct Grid2D {
    int nRows, nCols;
    std::vector<double> data;
    Grid2D(int nRows_, int nCols_) : nRows(nRows_), nCols(nCols_), data(static_cast<size_t>(nRows_) * nCols_) {}
    double& operator()(int i, int k) { return data[static_cast<size_t>(i) * nCols + k]; }
    double operator()(int i, int k) const { return data[static_cast<size_t>(i) * nCols + k]; }
};

int main() {
    // Contract: fixed-strike arithmetic-average call, continuously monitored,
    // I(t) = int_0^t S(u) du, A(t) = I(t)/t, payoff = (A(T) - K)^+.
    const double K = 100.0, r = 0.05, sigma = 0.20, T = 1.0, S0 = 100.0;

    const double Smax = 300.0, Amax = 300.0;
    const int NS = 120, NA = 240;
    const double dS = Smax / NS, dA = Amax / NA;
    const int i0 = static_cast<int>(std::round(S0 / dS));

    // Backward time-stepping runs from tau=0 (t=T) down to the last node before
    // t=0. The A-direction coefficient (S-A)/t is genuinely singular as t -> 0:
    // this is not a discretization artifact but a feature of the running-average
    // state variable itself (A(t) is degenerate, A(0)=S0, only for t>0 does it
    // carry a nontrivial conditional law). The economically meaningful price is
    // recovered as the t -> 0+ limit, which is approximated here by stopping one
    // step short, at t = dtau; the resulting O(dtau) approximation error is of
    // the same order as the scheme's own time-discretization error, so nothing is
    // lost asymptotically.
    const int M = 600;
    const double dtau = T / M;

    std::vector<double> S(NS + 1), A(NA + 1);
    for (int i = 0; i <= NS; ++i) S[i] = i * dS;
    for (int k = 0; k <= NA; ++k) A[k] = k * dA;

    // V(i,k) = V(S_i, A_k, current time level); Y is Stage 1's intermediate.
    // Both live in one contiguous buffer each (Grid2D) rather than NS+1
    // separately-allocated row vectors -- see the comment on Grid2D above.
    Grid2D V(NS + 1, NA + 1), Y(NS + 1, NA + 1);
    for (int i = 0; i <= NS; ++i)
        for (int k = 0; k <= NA; ++k) V(i, k) = std::max(A[k] - K, 0.0);

    // Thomas-solve scratch buffers, sized once to the larger of the two
    // sweep directions and reused every (k, then i) iteration of every time
    // step, instead of reallocating four vectors per call -- (NS+1)+(NA+1)
    // calls per step, times M-1 steps, made the allocator itself a
    // measurable share of the total cost before this change.
    std::vector<double> sub(std::max(NS, NA) + 1), diag(std::max(NS, NA) + 1),
        sup(std::max(NS, NA) + 1), rhs(std::max(NS, NA) + 1), sol(std::max(NS, NA) + 1);

    auto tStart = std::chrono::steady_clock::now();
    for (int n = 0; n < M - 1; ++n) {
        double t = T - (n + 1) * dtau;  // time level being solved for (t < T)

        // --- Stage 1: implicit in S, explicit in A, at every fixed A-row k ---
        for (int k = 0; k <= NA; ++k) {
            for (int i = 0; i <= NS; ++i) {
                double Si = S[i], sig2S2 = sigma * sigma * Si * Si;
                double alpha = dtau * (0.5 * sig2S2 / (dS * dS) - r * Si / (2.0 * dS));
                double gamma = dtau * (0.5 * sig2S2 / (dS * dS) + r * Si / (2.0 * dS));
                sub[i] = -alpha;
                sup[i] = -gamma;
                diag[i] = 1.0 + dtau * (sig2S2 / (dS * dS) + r);
                rhs[i] = V(i, k);
            }
            // S = 0: operator degenerates to identity (handled naturally by alpha=gamma=0).
            // S = Smax: zero-gradient (Neumann) closure. A Dirichlet value tied to the
            // vanilla-call asymptotic (V ~ S - Ke^{-r(T-t)}) would be wrong here: this
            // option's payoff depends on the running average A, not on S directly, so
            // there is no basis for asserting a specific value of V at large S independent
            // of A. Zero-gradient only asserts that V is locally linear in S near the
            // truncation boundary, which is the standard, model-agnostic closure.
            sub[NS] = -1.0;
            diag[NS] = 1.0;
            rhs[NS] = 0.0;
            thomasSolveInPlace(sub, diag, sup, rhs, sol, NS + 1);
            for (int i = 0; i <= NS; ++i) Y(i, k) = sol[i];
        }

        // --- Stage 2: implicit in A (upwinded), explicit in S, at every fixed S-row i ---
        for (int i = 0; i <= NS; ++i) {
            for (int k = 0; k <= NA; ++k) {
                // The valuation PDE ∂V/∂t + ν ∂V/∂A + ... = 0 (ν = (S-A)/t) is marched in
                // τ = T-t, giving ∂V/∂τ - ν ∂V/∂A = (...), i.e. an advection equation with
                // characteristic speed c = -ν. Upwinding must look in the direction the
                // characteristic came FROM: c > 0 (ν < 0) needs a backward difference,
                // c < 0 (ν >= 0) needs a forward difference -- the reverse of naively
                // upwinding on the sign of ν itself.
                double nu = (S[i] - A[k]) / t;
                double beta = dtau * nu / dA;
                if (nu >= 0.0) {                  // c = -nu <= 0: upwind = forward difference
                    sub[k] = 0.0;
                    diag[k] = 1.0 + beta;
                    sup[k] = -beta;
                } else {                           // c = -nu > 0: upwind = backward difference
                    sub[k] = beta;
                    diag[k] = 1.0 - beta;
                    sup[k] = 0.0;
                }
                rhs[k] = Y(i, k);
            }
            // Zero-gradient (Neumann) closure at both ends of the truncated A-range.
            diag[0] = 1.0; sup[0] = -1.0; rhs[0] = 0.0;
            sub[NA] = -1.0; diag[NA] = 1.0; rhs[NA] = 0.0;
            thomasSolveInPlace(sub, diag, sup, rhs, sol, NA + 1);
            for (int k = 0; k <= NA; ++k) V(i, k) = sol[k];
        }
    }
    auto tEnd = std::chrono::steady_clock::now();

    int k0 = static_cast<int>(std::round(S0 / dA));  // A0 = S0 at t = 0
    double priceADI = V(i0, k0);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Arithmetic Asian call (ADI, S0=A0=" << S0 << "): " << priceADI << "\n";
    std::cout << "Wall time for " << (M - 1) << " ADI time steps (" << (NS + 1) << "x" << (NA + 1)
               << " grid): " << std::setprecision(3)
               << std::chrono::duration<double, std::milli>(tEnd - tStart).count() << " ms\n";

    std::ofstream out("asian_option_value_surface.csv");
    out << "S,A,V\n";
    for (int i = 0; i <= NS; i += 2)
        for (int k = 0; k <= NA; k += 2)
            out << S[i] << "," << A[k] << "," << std::setprecision(8) << V(i, k) << "\n";
    out.close();

    return 0;
}
