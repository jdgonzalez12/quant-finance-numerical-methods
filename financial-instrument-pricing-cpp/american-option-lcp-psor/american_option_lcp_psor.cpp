// American put option pricing via the linear complementarity formulation of the
// Black-Scholes free-boundary problem, solved with Projected SOR (PSOR).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

double normalCdf(double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }

// Closed-form European put (Black-Scholes), used only as a lower-bound cross-check:
// the American put must dominate it by the early-exercise premium.
double europeanPutBS(double S0, double K, double r, double sigma, double T) {
    double d1 = (std::log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    double d2 = d1 - sigma * std::sqrt(T);
    return K * std::exp(-r * T) * normalCdf(-d2) - S0 * normalCdf(-d1);
}

// Brennan & Schwartz (1977) direct solver for the same linear complementarity
// problem PSOR solves iteratively. Because the tridiagonal operator here is an
// M-matrix for every dtau > 0 (diagonally dominant, non-positive off-diagonals
// -- see the comment on its construction below), the LCP's solution can be
// recovered in a *single* forward/backward sweep instead of an iterative
// relaxation: forward-eliminate exactly as in the unconstrained Thomas
// algorithm, then apply the obstacle projection max(.,payoff) directly during
// back-substitution rather than iterating a projected update to convergence.
// This is the same guarantee that makes projected Gauss-Seidel (PSOR) converge
// monotonically for an M-matrix LCP, specialized to the tridiagonal case where
// no iteration is needed at all: O(N) per time step, versus PSOR's O(N) per
// *iteration* times however many iterations relaxation happens to need.
std::vector<double> solveBrennanSchwartz(const std::vector<double>& sub,
                                          const std::vector<double>& diag,
                                          const std::vector<double>& super,
                                          const std::vector<double>& rhs,
                                          const std::vector<double>& obstacle) {
    int n = static_cast<int>(diag.size());
    std::vector<double> diagP(n), rhsP(n);
    diagP[0] = diag[0];
    rhsP[0] = rhs[0];
    for (int i = 1; i < n; ++i) {
        double m = sub[i] / diagP[i - 1];
        diagP[i] = diag[i] - m * super[i - 1];
        rhsP[i] = rhs[i] - m * rhsP[i - 1];
    }
    std::vector<double> V(n);
    V[n - 1] = std::max(rhsP[n - 1] / diagP[n - 1], obstacle[n - 1]);
    for (int i = n - 2; i >= 0; --i)
        V[i] = std::max((rhsP[i] - super[i] * V[i + 1]) / diagP[i], obstacle[i]);
    return V;
}

int main() {
    // Contract and market data (chosen to coincide with the Longstaff-Schwartz
    // Monte Carlo benchmark already computed elsewhere in this repository, so the
    // two prices -- obtained by entirely different numerical methods -- can be
    // cross-checked against each other).
    const double K = 40.0;
    const double r = 0.06;
    const double sigma = 0.20;
    const double T = 1.0;
    const double S0 = 36.0;

    // Spatial grid: S_max chosen well past K so V(S_max, .) == 0 is an accurate
    // Dirichlet closure; S0 is made to land exactly on a grid node.
    const double Smax = 180.0;
    const int N = 360;              // S0 = 36 -> node index 72
    const double dS = Smax / N;
    const int i0 = static_cast<int>(std::round(S0 / dS));

    // Time grid. Implicit (backward) Euler is used deliberately rather than
    // Crank-Nicolson: the discrete operator below is an M-matrix (diagonally
    // dominant with non-positive off-diagonals) for every dtau > 0, which gives a
    // discrete maximum principle and hence a monotone, oscillation-free solution
    // near the non-smooth kink of max(K-S,0). Crank-Nicolson does not have this
    // property uniformly in the mesh ratio and is well known (Rannacher, 1984) to
    // produce spurious oscillations from a non-smooth initial condition unless the
    // first few steps are damped separately; backward Euler avoids that
    // complication at the cost of only first-order accuracy in time, which is
    // compensated here with a fine time grid.
    const int M = 2000;
    const double dtau = T / M;

    std::vector<double> S(N + 1), payoff(N + 1), V(N + 1);
    for (int i = 0; i <= N; ++i) {
        S[i] = i * dS;
        payoff[i] = std::max(K - S[i], 0.0);
    }
    V = payoff;  // terminal condition at tau = 0 (i.e. t = T)

    // Tridiagonal coefficients of (I - dtau * L) V^{n+1} = V^n, where L is the
    // Black-Scholes spatial operator L V = 1/2 sigma^2 S^2 V_SS + r S V_S - r V,
    // discretized with standard central differences on the interior nodes.
    std::vector<double> subDiag(N + 1, 0.0), diag(N + 1, 0.0), superDiag(N + 1, 0.0);
    for (int i = 1; i < N; ++i) {
        double Si = S[i];
        double sig2S2 = sigma * sigma * Si * Si;
        double alpha = dtau * (0.5 * sig2S2 / (dS * dS) - r * Si / (2.0 * dS));
        double gamma = dtau * (0.5 * sig2S2 / (dS * dS) + r * Si / (2.0 * dS));
        subDiag[i] = -alpha;
        superDiag[i] = -gamma;
        diag[i] = 1.0 + dtau * (sig2S2 / (dS * dS) + r);
    }
    // Dirichlet rows: V(0,t) = K (the put's intrinsic value; S = 0 is absorbing
    // under geometric Brownian motion, so immediate exercise is optimal there and
    // remains so for all t), V(Smax,t) = 0 (deep out-of-the-money truncation).
    diag[0] = 1.0;
    diag[N] = 1.0;

    // Lower obstacle for the LCP: V >= payoff everywhere, with equality enforced
    // (trivially, since payoff(0) = K and payoff(N) = 0) at the two boundary rows.
    const std::vector<double> &c = payoff;

    const double omega = 1.5;       // SOR relaxation parameter, 1 < omega < 2
    const double tol = 1e-9;
    const int maxPsorIter = 10000;

    long long totalPsorIters = 0;
    auto tPsorStart = std::chrono::steady_clock::now();
    std::vector<double> rhs(N + 1);
    for (int n = 0; n < M; ++n) {
        rhs = V;
        rhs[0] = K;
        rhs[N] = 0.0;

        // Projected SOR: Gauss-Seidel sweep with a per-component projection onto
        // the constraint x_j >= c_j, applied after every relaxed update -- the
        // discrete analogue of the complementarity condition
        // (x - c)^T (A x - b) = 0, x >= c, A x >= b.
        for (int iter = 0; iter < maxPsorIter; ++iter) {
            double maxDelta = 0.0;
            for (int j = 0; j <= N; ++j) {
                double sum = rhs[j];
                if (j > 0) sum -= subDiag[j] * V[j - 1];
                if (j < N) sum -= superDiag[j] * V[j + 1];
                double y = sum / diag[j];
                double vOld = V[j];
                double vNew = std::max(c[j], vOld + omega * (y - vOld));
                V[j] = vNew;
                maxDelta = std::max(maxDelta, std::fabs(vNew - vOld));
            }
            ++totalPsorIters;
            if (maxDelta < tol) break;
        }
    }
    auto tPsorEnd = std::chrono::steady_clock::now();

    double priceLCP = V[i0];
    double priceEuropean = europeanPutBS(S0, K, r, sigma, T);

    // Independent end-to-end run of the same LCP with the direct Brennan-Schwartz
    // solver (one forward/backward sweep per time step, no relaxation), for a
    // genuine cross-check of both the numerics and the latency claim.
    std::vector<double> Vbs = payoff;
    auto tBsStart = std::chrono::steady_clock::now();
    for (int n = 0; n < M; ++n) {
        rhs = Vbs;
        rhs[0] = K;
        rhs[N] = 0.0;
        Vbs = solveBrennanSchwartz(subDiag, diag, superDiag, rhs, payoff);
    }
    auto tBsEnd = std::chrono::steady_clock::now();

    double priceBS = Vbs[i0];
    double maxDiffPsorBs = 0.0;
    for (int i = 0; i <= N; ++i) maxDiffPsorBs = std::max(maxDiffPsorBs, std::fabs(V[i] - Vbs[i]));

    double psorMs = std::chrono::duration<double, std::milli>(tPsorEnd - tPsorStart).count();
    double bsMs = std::chrono::duration<double, std::milli>(tBsEnd - tBsStart).count();

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "American put (LCP/PSOR), S0=" << S0 << ": " << priceLCP << "\n";
    std::cout << "American put (LCP/Brennan-Schwartz), S0=" << S0 << ": " << priceBS << "\n";
    std::cout << "European put (Black-Scholes closed form):  " << priceEuropean << "\n";
    std::cout << "Early-exercise premium: " << (priceLCP - priceEuropean) << "\n\n";
    std::cout << "Cross-check |V_PSOR - V_BrennanSchwartz| over the whole grid: "
               << std::scientific << maxDiffPsorBs << std::fixed << "\n\n";
    std::cout << "Total PSOR relaxation sweeps over " << M << " time steps: " << totalPsorIters
               << "  (" << std::setprecision(2) << (double(totalPsorIters) / M)
               << " avg per step, vs. Brennan-Schwartz's exactly 1 sweep per step)\n";
    std::cout << std::setprecision(3) << "Wall time: PSOR = " << psorMs
               << " ms,  Brennan-Schwartz = " << bsMs << " ms  (" << (psorMs / bsMs)
               << "x)\n";

    std::ofstream out("american_put_value.csv");
    out << std::fixed << std::setprecision(8);
    out << "S,V_american,V_american_bs,V_european_intrinsic\n";
    for (int i = 0; i <= N; ++i) {
        if (S[i] > 3 * K) break;  // only export the economically relevant range
        out << S[i] << "," << V[i] << "," << Vbs[i] << "," << payoff[i] << "\n";
    }
    out.close();

    return 0;
}
