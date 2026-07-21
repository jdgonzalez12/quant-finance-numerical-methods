// Keller's Box scheme for the Black-Scholes PDE (Keller, 1971).
//
// Crank-Nicolson discretizes V_tau and the spatial derivatives of V at the
// same mesh points, which is what produces its well-documented spurious
// oscillations near the strike for non-smooth (kinked) payoffs. The Box
// scheme sidesteps this by first reducing the second-order PDE to a
// first-order system, introducing the gradient W = dV/dS as an independent
// unknown, and then discretizing *that* system on a box spanning one mesh
// cell in S and one time step in tau, averaging every quantity to the center
// of the box. Both V and W end up second-order accurate, and the resulting
// scheme is unconditionally stable for the same reason Crank-Nicolson is
// (it is, in fact, a reformulation of an implicit trapezoidal rule, just
// carried out on a first-order system rather than the original second-order
// equation).
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace std;

// Reference-only dense solver (Gaussian elimination with partial pivoting).
// Used exclusively to cross-validate solveBoxBlockBidiagonal below; O(n^3)
// and never called inside the per-timestep hot path.
vector<double> solveDenseReference(vector<vector<double>> A, vector<double> b) {
    int n = static_cast<int>(b.size());
    for (int col = 0; col < n; ++col) {
        int piv = col;
        for (int row = col + 1; row < n; ++row)
            if (fabs(A[row][col]) > fabs(A[piv][col])) piv = row;
        swap(A[col], A[piv]);
        swap(b[col], b[piv]);
        for (int row = col + 1; row < n; ++row) {
            double m = A[row][col] / A[col][col];
            if (m == 0.0) continue;
            for (int k = col; k < n; ++k) A[row][k] -= m * A[col][k];
            b[row] -= m * b[col];
        }
    }
    vector<double> x(n);
    for (int row = n - 1; row >= 0; --row) {
        double s = b[row];
        for (int k = row + 1; k < n; ++k) s -= A[row][k] * x[k];
        x[row] = s / A[row][row];
    }
    return x;
}

// Per-cell coefficients of the Box scheme's two equations (P: box average of
// dV/dS = W; Q: box average of the PDE), for cells j = 0, ..., J-1, plus the
// two Dirichlet values that close the system at S=0 and S=Smax.
struct BoxSystem {
    vector<double> p00, p01, p10, p11, pr;  // P_j: p00 V_j + p01 W_j + p10 V_{j+1} + p11 W_{j+1} = pr
    vector<double> q00, q01, q10, q11, qr;  // Q_j: analogous
    double V0, VJ;
};

// O(J) direct solve exploiting the Box scheme's exact structure: grouping
// unknowns as x_j = (V_j, W_j), the system is block-*bidiagonal* (each cell's
// two equations couple only x_j and x_{j+1} -- there is no direct coupling to
// x_{j-1}, since the Box scheme is a first-order system, not the
// central-difference stencil that would give block-*tridiagonal* coupling),
// bordered by two scalar Dirichlet rows fixing V_0 and V_J. This is solved by
// carrying a single reduced equation R_j: e0 V_j + e1 W_j = er forward
// (seeded from V_0 = 0), combining it algebraically with cell j's two
// equations to eliminate (V_j, W_j) and produce R_{j+1}, then recovering
// every (V_j, W_j) by back-substitution once the last reduced equation meets
// the V_J boundary condition. This is the 2x2-block generalization of the
// scalar Thomas algorithm used throughout this repository (see
// tridiagonal-solvers/), and costs O(J) instead of the O(J^3) a dense solve
// of the same 2(J+1)-dimensional system requires -- see the "Computational
// complexity" section of this folder's README for the exact operation count.
pair<vector<double>, vector<double>> solveBoxBlockBidiagonal(const BoxSystem& sys) {
    int J = static_cast<int>(sys.p00.size());

    vector<double> e0(J + 1), e1(J + 1), er(J + 1);
    vector<double> A1(J), B1(J), C1(J), D1(J);

    e0[0] = 1.0;
    e1[0] = 0.0;
    er[0] = sys.V0;

    const double kMinPivot = 1e-12;
    for (int j = 0; j < J; ++j) {
        double E0 = e0[j], E1 = e1[j], ER = er[j];
        if (fabs(E0) < kMinPivot)
            throw runtime_error("solveBoxBlockBidiagonal: degenerate pivot at cell " + to_string(j));

        // Eliminate V_j = (ER - E1*W_j)/E0 from P_j and Q_j.
        double a1 = sys.p01[j] - sys.p00[j] * E1 / E0;
        double b1 = sys.p10[j];
        double c1 = sys.p11[j];
        double d1 = sys.pr[j] - sys.p00[j] * ER / E0;

        double a2 = sys.q01[j] - sys.q00[j] * E1 / E0;
        double b2 = sys.q10[j];
        double c2 = sys.q11[j];
        double d2 = sys.qr[j] - sys.q00[j] * ER / E0;

        A1[j] = a1;
        B1[j] = b1;
        C1[j] = c1;
        D1[j] = d1;

        if (fabs(a1) < kMinPivot)
            throw runtime_error("solveBoxBlockBidiagonal: degenerate W_j pivot at cell " + to_string(j));

        // Eliminate W_j between (a1,b1,c1,d1) and (a2,b2,c2,d2):
        //   a2*(P') - a1*(Q') removes W_j, leaving R_{j+1} in (V_{j+1}, W_{j+1}).
        e0[j + 1] = a2 * b1 - a1 * b2;
        e1[j + 1] = a2 * c1 - a1 * c2;
        er[j + 1] = a2 * d1 - a1 * d2;
    }

    if (fabs(e1[J]) < kMinPivot)
        throw runtime_error("solveBoxBlockBidiagonal: degenerate terminal pivot");

    vector<double> V(J + 1), W(J + 1);
    V[J] = sys.VJ;
    W[J] = (er[J] - e0[J] * sys.VJ) / e1[J];

    for (int j = J - 1; j >= 0; --j) {
        W[j] = (D1[j] - B1[j] * V[j + 1] - C1[j] * W[j + 1]) / A1[j];
        V[j] = (er[j] - e1[j] * W[j]) / e0[j];
    }
    return {V, W};
}

double blackScholesCall(double S, double K, double r, double sigma, double T) {
    if (T <= 0.0) return max(S - K, 0.0);
    double d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
    double d2 = d1 - sigma * sqrt(T);
    auto Phi = [](double x) { return 0.5 * erfc(-x / sqrt(2.0)); };
    return S * Phi(d1) - K * exp(-r * T) * Phi(d2);
}

// Runs the full time-stepping simulation with the chosen solver (fast block
// solve, or the O(n^3) dense reference kept for cross-validation).
pair<vector<double>, vector<double>> runSimulation(int J, int N, double Smax, double K,
                                                    double sigma, double r, double T,
                                                    bool useDenseReference) {
    const double h = Smax / J;
    const double dtau = T / N;

    vector<double> S(J + 1);
    for (int j = 0; j <= J; ++j) S[j] = j * h;

    vector<double> V(J + 1), W(J + 1, 0.0);
    for (int j = 0; j <= J; ++j) V[j] = max(S[j] - K, 0.0);
    for (int j = 0; j < J; ++j) W[j] = (V[j + 1] - V[j]) / h;
    W[J] = W[J - 1];

    for (int step = 0; step < N; ++step) {
        double tau_new = (step + 1) * dtau;

        BoxSystem sys;
        sys.p00.resize(J); sys.p01.resize(J); sys.p10.resize(J); sys.p11.resize(J); sys.pr.resize(J);
        sys.q00.resize(J); sys.q01.resize(J); sys.q10.resize(J); sys.q11.resize(J); sys.qr.resize(J);
        sys.V0 = 0.0;
        sys.VJ = Smax - K * exp(-r * tau_new);

        for (int j = 0; j < J; ++j) {
            double Sm = 0.5 * (S[j] + S[j + 1]);
            double diffCoef = 0.5 * sigma * sigma * Sm * Sm / h;
            double Vold_mid = 0.5 * (V[j] + V[j + 1]);

            sys.p00[j] = -1.0 / h;
            sys.p10[j] = 1.0 / h;
            sys.p01[j] = -0.5;
            sys.p11[j] = -0.5;
            sys.pr[j] = 0.0;

            sys.q00[j] = -1.0 / (2.0 * dtau) - r / 2.0;
            sys.q10[j] = -1.0 / (2.0 * dtau) - r / 2.0;
            sys.q01[j] = -diffCoef + r * Sm / 2.0;
            sys.q11[j] = diffCoef + r * Sm / 2.0;
            sys.qr[j] = -Vold_mid / dtau;
        }

        if (useDenseReference) {
            int n = 2 * (J + 1);
            vector<vector<double>> A(n, vector<double>(n, 0.0));
            vector<double> b(n, 0.0);
            A[0][0] = 1.0;
            b[0] = sys.V0;
            for (int j = 0; j < J; ++j) {
                int iVj = 2 * j, iWj = 2 * j + 1, iVj1 = 2 * (j + 1), iWj1 = 2 * (j + 1) + 1;
                int row1 = 2 * j + 1, row2 = 2 * j + 2;
                A[row1][iVj] = sys.p00[j]; A[row1][iVj1] = sys.p10[j];
                A[row1][iWj] = sys.p01[j]; A[row1][iWj1] = sys.p11[j];
                b[row1] = sys.pr[j];
                A[row2][iVj] = sys.q00[j]; A[row2][iVj1] = sys.q10[j];
                A[row2][iWj] = sys.q01[j]; A[row2][iWj1] = sys.q11[j];
                b[row2] = sys.qr[j];
            }
            A[n - 1][2 * J] = 1.0;
            b[n - 1] = sys.VJ;
            vector<double> x = solveDenseReference(A, b);
            for (int j = 0; j <= J; ++j) {
                V[j] = x[2 * j];
                W[j] = x[2 * j + 1];
            }
        } else {
            auto [Vnew, Wnew] = solveBoxBlockBidiagonal(sys);
            V = Vnew;
            W = Wnew;
        }
    }
    return {V, W};
}

int main() {
    const double K = 25.0, Smax = 50.0, sigma = 0.20, r = 0.0436, T = 1.0;
    const int J = 200, N = 100;
    const double h = Smax / J;

    vector<double> S(J + 1);
    for (int j = 0; j <= J; ++j) S[j] = j * h;

    auto [V, W] = runSimulation(J, N, Smax, K, sigma, r, T, /*useDenseReference=*/false);

    cout << fixed << setprecision(6);
    cout << "Keller Box scheme (O(J) block-bidiagonal solve), K=" << K << " Smax=" << Smax
         << " sigma=" << sigma << " r=" << r << " T=" << T << " (J=" << J << " cells, N=" << N
         << " steps)\n\n";
    cout << setw(8) << "S" << setw(16) << "V (Box)" << setw(16) << "V (closed form)"
         << setw(16) << "|error|" << setw(16) << "W = dV/dS" << "\n";
    for (int j = 0; j <= J; j += J / 10) {
        double exact = blackScholesCall(max(S[j], 1e-8), K, r, sigma, T);
        double err = fabs(V[j] - exact);
        cout << setw(8) << S[j] << setw(16) << V[j] << setw(16) << exact << setw(16) << err
             << setw(16) << W[j] << "\n";
    }
    double maxErrAll = 0.0;
    for (int j = 1; j < J; ++j) {
        double exact = blackScholesCall(S[j], K, r, sigma, T);
        maxErrAll = max(maxErrAll, fabs(V[j] - exact));
    }
    cout << "\nmax |V_Box - V_closed_form| over interior nodes = " << maxErrAll << "\n";

    // Cross-validation: an entirely independent O(n^3) dense solve of the
    // identical per-step linear system, run end-to-end, compared against the
    // O(J) result above. This is the same "two independent implementations,
    // checked against each other" discipline used throughout this repository
    // (e.g. tridiagonal-solvers/'s LU vs. Godunov double sweep).
    auto [Vref, Wref] = runSimulation(J, N, Smax, K, sigma, r, T, /*useDenseReference=*/true);
    double maxDiffV = 0.0, maxDiffW = 0.0;
    for (int j = 0; j <= J; ++j) {
        maxDiffV = max(maxDiffV, fabs(V[j] - Vref[j]));
        maxDiffW = max(maxDiffW, fabs(W[j] - Wref[j]));
    }
    cout << "\nCross-check vs. an independent O(n^3) dense solve of the identical system:\n";
    cout << "  max |V_fast - V_dense| = " << scientific << maxDiffV << "\n";
    cout << "  max |W_fast - W_dense| = " << maxDiffW << fixed << "\n";

    return 0;
}
