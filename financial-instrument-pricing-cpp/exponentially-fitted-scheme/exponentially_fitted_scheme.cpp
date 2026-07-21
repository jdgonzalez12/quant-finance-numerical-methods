// Exponentially fitted finite-difference schemes (Il'in, 1969; Duffy, 1980)
// for convection-dominated convection-diffusion problems, of which
// Black-Scholes is a special case.
//
// A standard centered-difference discretization of
//   sigma*u'' + mu*u' = 0
// is only conditionally well-behaved: once the mesh (cell) Peclet number
//   Pe_h = mu*h / (2*sigma)
// exceeds 1, the centered scheme's amplification factors change sign between
// neighboring nodes and the discrete solution oscillates around the true
// (monotone) solution, even though the scheme remains linearly stable in the
// usual sense. The fix is to replace the bare second-difference operator by
// one scaled with an exponential fitting factor
//   rho(Pe_h) = Pe_h * coth(Pe_h)
// chosen so that the resulting difference equation reproduces the *exact*
// solution of the constant-coefficient problem at every mesh point,
// regardless of h. This file first reproduces that exact result on the
// constant-coefficient ODE, then carries the same fitted operator into the
// convection-dominated regime of the Black-Scholes PDE.
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <utility>
#include <vector>

using namespace std;

// Thomas algorithm for Au = f with a[j]u_{j-1}+b[j]u_j+c[j]u_{j+1}=f[j].
// a, c are read-only (taken by const reference); b, f are consumed in place
// into caller-provided scratch buffers so no vector is copied or freed on
// this routine's own account -- it is called once per time step here.
void solveTridiagonalInPlace(const vector<double>& a, vector<double>& b,
                              const vector<double>& c, vector<double>& f, vector<double>& u) {
    int n = static_cast<int>(b.size());
    for (int j = 1; j < n; ++j) {
        double m = a[j] / b[j - 1];
        b[j] -= m * c[j - 1];
        f[j] -= m * f[j - 1];
    }
    u[n - 1] = f[n - 1] / b[n - 1];
    for (int j = n - 2; j >= 0; --j) u[j] = (f[j] - c[j] * u[j + 1]) / b[j];
}

double cothStable(double x) {
    // coth(x) = (e^{2x}+1)/(e^{2x}-1). Two limits need care, both of which
    // arise across the range of cell Peclet numbers seen in Part B (Pe -> 0
    // as S -> Smax where convection is weak relative to diffusion, and
    // Pe -> infinity as S -> 0 where it is not):
    //   x -> 0: naive evaluation is 0/0 -- use the Laurent series 1/x + x/3.
    //   |x| large: e^{2x} overflows to +inf in IEEE double once x gtr ~355,
    //   giving inf/inf = NaN, even though coth(x) -> sign(x) exactly in that
    //   regime (the true function has no singularity there at all).
    if (fabs(x) < 1e-8) return 1.0 / x + x / 3.0;
    if (x > 20.0) return 1.0;
    if (x < -20.0) return -1.0;
    return (exp(2 * x) + 1) / (exp(2 * x) - 1);
}

// ---------------------------------------------------------------------
// Part A: sigma*u'' + mu*u' = 0 on (A,B), u(A)=beta0, u(B)=beta1.
// Exact solution: u(x) = beta0 + (beta1-beta0) * (1-exp(-mu*(x-A)/sigma)) /
//                                                 (1-exp(-mu*(B-A)/sigma)).
// ---------------------------------------------------------------------
void partA() {
    const double sigma = 0.01, mu = 1.0, A = 0.0, B = 1.0, beta0 = 0.0, beta1 = 1.0;
    const int J = 20;
    const double h = (B - A) / J;
    const double Pe = mu * h / (2 * sigma);  // cell Peclet number

    vector<double> x(J + 1);
    for (int j = 0; j <= J; ++j) x[j] = A + j * h;

    auto exact = [&](double xx) {
        return beta0 + (beta1 - beta0) * (1 - exp(-mu * (xx - A) / sigma)) /
                           (1 - exp(-mu * (B - A) / sigma));
    };

    // Standard centered scheme (rho = 1):
    //   sigma*(U_{j-1}-2U_j+U_{j+1})/h^2 + mu*(U_{j+1}-U_{j-1})/(2h) = 0
    int n = J - 1;
    vector<double> a(n), b(n), c(n), f(n, 0.0);
    for (int j = 1; j <= n; ++j) {
        a[j - 1] = sigma / (h * h) - mu / (2 * h);
        b[j - 1] = -2 * sigma / (h * h);
        c[j - 1] = sigma / (h * h) + mu / (2 * h);
    }
    f[0] -= a[0] * beta0;
    f[n - 1] -= c[n - 1] * beta1;
    vector<double> Ucentered(n);
    solveTridiagonalInPlace(a, b, c, f, Ucentered);

    // Exponentially fitted scheme:
    //   sigma*rho*(U_{j-1}-2U_j+U_{j+1})/h^2 + mu*(U_{j+1}-U_{j-1})/(2h) = 0
    double rho = cothStable(Pe) * Pe;
    vector<double> af(n), bf(n), cf(n), ff(n, 0.0);
    for (int j = 1; j <= n; ++j) {
        af[j - 1] = sigma * rho / (h * h) - mu / (2 * h);
        bf[j - 1] = -2 * sigma * rho / (h * h);
        cf[j - 1] = sigma * rho / (h * h) + mu / (2 * h);
    }
    ff[0] -= af[0] * beta0;
    ff[n - 1] -= cf[n - 1] * beta1;
    vector<double> Ufitted(n);
    solveTridiagonalInPlace(af, bf, cf, ff, Ufitted);

    cout << fixed << setprecision(8);
    cout << "Part A: sigma*u'' + mu*u' = 0, sigma=" << sigma << " mu=" << mu << " (J=" << J
         << ", cell Peclet number = " << Pe << ")\n\n";
    cout << setw(8) << "x" << setw(14) << "exact" << setw(16) << "centered"
         << setw(16) << "|err| centered" << setw(14) << "fitted" << setw(16)
         << "|err| fitted" << "\n";
    double maxErrC = 0.0, maxErrF = 0.0;
    for (int j = 0; j <= J; ++j) {
        double ex = exact(x[j]);
        double uc = (j == 0) ? beta0 : (j == J ? beta1 : Ucentered[j - 1]);
        double uf = (j == 0) ? beta0 : (j == J ? beta1 : Ufitted[j - 1]);
        maxErrC = max(maxErrC, fabs(uc - ex));
        maxErrF = max(maxErrF, fabs(uf - ex));
        cout << setw(8) << x[j] << setw(14) << ex << setw(16) << uc << setw(16)
             << fabs(uc - ex) << setw(14) << uf << setw(16) << fabs(uf - ex) << "\n";
    }
    cout << "\nmax |error|: centered = " << maxErrC << ",  fitted = " << maxErrF
         << "  (fitted is exact at mesh points up to floating-point round-off,\n"
         << "   the classical Il'in result for constant-coefficient problems)\n\n";

    // Monotonicity check: a physically meaningful solution here is monotone
    // (u ranges from 0 to 1 with mu>0, sigma>0 and no interior extremum);
    // count sign changes in consecutive differences as a proxy for spurious
    // oscillation.
    auto countSignChanges = [](const vector<double>& v) {
        int count = 0;
        for (size_t i = 2; i < v.size(); ++i) {
            double d1 = v[i - 1] - v[i - 2], d2 = v[i] - v[i - 1];
            if (d1 * d2 < 0) ++count;
        }
        return count;
    };
    vector<double> fullC(J + 1), fullF(J + 1);
    for (int j = 0; j <= J; ++j) {
        fullC[j] = (j == 0) ? beta0 : (j == J ? beta1 : Ucentered[j - 1]);
        fullF[j] = (j == 0) ? beta0 : (j == J ? beta1 : Ufitted[j - 1]);
    }
    cout << "Sign changes in consecutive differences (0 = monotone, >0 = oscillatory): "
         << "centered = " << countSignChanges(fullC) << ", fitted = " << countSignChanges(fullF)
         << "\n\n";
}

// ---------------------------------------------------------------------
// Part B: same fitting idea applied to the Black-Scholes PDE in a
// convection-dominated regime (low sigma, high r -> large local cell Peclet
// number near large S), fully implicit in time.
// ---------------------------------------------------------------------
void partB() {
    const double K = 25.0, Smax = 100.0, sigma = 0.01, r = 0.50, T = 0.25;
    const int J = 120, N = 50;
    const int reportAfterSteps = 2;  // oscillation from the payoff kink is
                                      // strongest right after it is introduced;
                                      // implicit time-stepping's own numerical
                                      // diffusion damps it out over many steps.
    const double h = Smax / J, dtau = T / N;

    vector<double> S(J + 1);
    for (int j = 0; j <= J; ++j) S[j] = j * h;

    // Runs the full N-step time march in a single pass, snapshotting the grid
    // at `reportAfterSteps` along the way instead of re-running the first
    // `reportAfterSteps` steps twice (once standalone, once again as a
    // prefix of the full N-step run) -- halving the number of tridiagonal
    // solves this function performs. The per-step coefficient and Thomas-solve
    // buffers are allocated once, outside the time loop, and reused every
    // step rather than reallocated N times.
    auto runScheme = [&](bool fitted, int stepsToRun, int snapshotAt) {
        vector<double> V(J + 1);
        for (int j = 0; j <= J; ++j) V[j] = max(S[j] - K, 0.0);
        vector<double> Vsnapshot;

        int n = J - 1;
        vector<double> aImp(n), bImp(n), cImp(n), fRHS(n), Vnew(n);
        for (int step = 0; step < stepsToRun; ++step) {
            double tau = (step + 1) * dtau;
            // Crank-Nicolson: (I - dtau/2 L) V^{n+1} = (I + dtau/2 L) V^n,
            // where L is the (fitted or centered) spatial operator. This is
            // the scheme the book explicitly attributes the oscillation
            // problem to (Ch. 17.4) -- unlike backward Euler, its implicit
            // half-weight is not enough to force diagonal dominance on the
            // centered operator once the cell Peclet number is large.
            for (int idx = 0; idx < n; ++idx) {
                int j = idx + 1;
                double sig2 = 0.5 * sigma * sigma * S[j] * S[j];
                double mu = r * S[j];
                double Pe = (sig2 > 1e-14) ? (mu * h) / (2 * sig2) : 0.0;
                double rho = fitted ? cothStable(Pe) * Pe : 1.0;
                double diff = sig2 * rho / (h * h);

                double subL = diff - mu / (2 * h);
                double supL = diff + mu / (2 * h);
                double diagL = -2 * diff - r;

                aImp[idx] = -0.5 * dtau * subL;
                cImp[idx] = -0.5 * dtau * supL;
                bImp[idx] = 1.0 - 0.5 * dtau * diagL;

                double explicitTerm = 0.5 * dtau * diagL * V[j];
                if (j > 0) explicitTerm += 0.5 * dtau * subL * V[j - 1];
                if (j < J) explicitTerm += 0.5 * dtau * supL * V[j + 1];
                fRHS[idx] = V[j] + explicitTerm;
            }
            // V(0,*)=0 at both time levels, so the j=1 equation needs no extra
            // boundary correction beyond the V[j-1]=0 already used above.
            double VmaxNew = Smax - K * exp(-r * tau);
            fRHS[n - 1] += -cImp[n - 1] * VmaxNew;  // implicit-level boundary term
            // (the explicit-level boundary term used V[J]=VmaxOld directly,
            // since V[J] is only overwritten with VmaxNew after this solve)

            solveTridiagonalInPlace(aImp, bImp, cImp, fRHS, Vnew);
            for (int idx = 0; idx < n; ++idx) V[idx + 1] = Vnew[idx];
            V[0] = 0.0;
            V[J] = VmaxNew;

            if (step + 1 == snapshotAt) Vsnapshot = V;
        }
        return make_pair(Vsnapshot, V);
    };

    auto [Vcentered, VcenteredFinal] = runScheme(false, N, reportAfterSteps);
    auto [Vfitted, VfittedFinal] = runScheme(true, N, reportAfterSteps);

    auto countSignChanges = [](const vector<double>& v, int lo, int hi) {
        int count = 0;
        for (int i = lo + 1; i < hi; ++i) {
            double d1 = v[i] - v[i - 1], d2 = v[i + 1] - v[i];
            if (d1 * d2 < 0) ++count;
        }
        return count;
    };
    // Look for oscillation in the region around the kink S=K, where the
    // payoff's discontinuous second derivative most strongly excites it.
    int jK = static_cast<int>(round(K / h));
    int lo = max(1, jK - 12), hi = min(J - 1, jK + 12);

    cout << "Part B: Black-Scholes, sigma=" << sigma << " r=" << r
         << " (convection-dominated: cell Peclet number at S=Smax is "
         << (r * Smax * h) / (sigma * sigma * Smax * Smax) << ")\n\n";
    cout << "After " << reportAfterSteps << " implicit time step(s) (the payoff kink has just "
         << "been imposed and not yet smoothed away):\n";
    cout << setw(8) << "S" << setw(16) << "V (centered)" << setw(16) << "V (fitted)" << "\n";
    for (int j = lo; j <= hi; ++j)
        cout << setw(8) << S[j] << setw(16) << Vcentered[j] << setw(16) << Vfitted[j] << "\n";
    cout << "\nSign changes near the kink S=K (region [" << S[lo] << "," << S[hi]
         << "]): centered = " << countSignChanges(Vcentered, lo, hi)
         << ", fitted = " << countSignChanges(Vfitted, lo, hi) << "\n";

    cout << "\nAfter all " << N << " steps (T=" << T << "), for comparison:\n";
    cout << setw(8) << "S" << setw(16) << "V (centered)" << setw(16) << "V (fitted)" << "\n";
    for (int j = lo; j <= hi; ++j)
        cout << setw(8) << S[j] << setw(16) << VcenteredFinal[j] << setw(16) << VfittedFinal[j]
             << "\n";
    cout << "Sign changes near the kink at T: centered = "
         << countSignChanges(VcenteredFinal, lo, hi)
         << ", fitted = " << countSignChanges(VfittedFinal, lo, hi) << "\n";

    // The book notes the oscillation is "more pronounced in the delta" than
    // in the price itself (a non-smooth payoff's second derivative is what
    // actually excites it) -- so check the discrete delta dV/dS for spurious
    // non-monotonicity too, not just the price level.
    auto delta = [&](const vector<double>& V) {
        vector<double> d(J + 1, 0.0);
        for (int j = 1; j < J; ++j) d[j] = (V[j + 1] - V[j - 1]) / (2 * h);
        return d;
    };
    vector<double> deltaC = delta(VcenteredFinal), deltaF = delta(VfittedFinal);
    cout << "\nDiscrete delta dV/dS near the kink:\n";
    cout << setw(8) << "S" << setw(16) << "delta (centered)" << setw(16) << "delta (fitted)"
         << "\n";
    for (int j = lo; j <= hi; ++j)
        cout << setw(8) << S[j] << setw(16) << deltaC[j] << setw(16) << deltaF[j] << "\n";
    cout << "Sign changes in delta near the kink: centered = "
         << countSignChanges(deltaC, lo, hi) << ", fitted = " << countSignChanges(deltaF, lo, hi)
         << "\n";
}

int main() {
    partA();
    partB();
    return 0;
}
