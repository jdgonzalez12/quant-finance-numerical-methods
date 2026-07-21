# Numerical Methods

Python/Jupyter and C++ implementations of numerical methods spanning classical numerical analysis,
stochastic simulation, and quantitative finance: root-finding, numerical linear algebra, curve
fitting and interpolation, finite differences, ODE and SDE solvers, finite-difference PDE pricing
(Black-Scholes and Heston), Monte Carlo and variance-reduction techniques, sensitivity (Greeks)
estimation, risk measurement, portfolio construction (Bayesian view combination, semidefinite
correlation repair), constrained and large-scale first-order/quasi-Newton optimization, sequential
Bayesian filtering, and a dedicated C++ track for finite-difference and complementarity methods.

Every algorithm is implemented from first principles (no `scipy.optimize`, `numpy.linalg`, or
off-the-shelf solvers for the core method being studied), and every notebook validates its result
against a closed-form solution, a published reference value, or by comparing multiple independent
methods against each other.

## Projects

| Folder | Covers |
|---|---|
| [`root-finding-and-monte-carlo/`](root-finding-and-monte-carlo/) | Root-finding methods (bisection, false position, fixed-point, Newton-Raphson) and Monte Carlo integration |
| [`numerical-linear-algebra-and-curve-fitting/`](numerical-linear-algebra-and-curve-fitting/) | Numerical linear algebra (determinant, inverse, LU, Gram-Schmidt, linear systems), curve fitting, and interpolation |
| [`finite-differences-integration-odes/`](finite-differences-integration-odes/) | Finite-difference differentiation, numerical integration, and ODE solvers (Euler, midpoint, Heun) |
| [`crank-nicolson-black-scholes/`](crank-nicolson-black-scholes/) | European call option pricing under Black-Scholes via a Crank-Nicolson finite-difference scheme |
| [`stochastic-differential-equations/`](stochastic-differential-equations/) | Euler-Maruyama (well-posedness, strong/weak order, mean-square stability), Milstein, derivative-free stochastic Runge-Kutta, implicit $\theta$-schemes for stiff systems, multidimensional Milstein under non-commutative noise (the Lévy area), and the order-2.0 weak Taylor scheme |
| [`crank-nicolson-heston/`](crank-nicolson-heston/) | European option pricing under the Heston stochastic-volatility model via ADI (Douglas) finite differences |
| [`longstaff-schwartz-lsmc/`](longstaff-schwartz-lsmc/) | American option pricing via least-squares Monte Carlo |
| [`quasi-monte-carlo/`](quasi-monte-carlo/) | Sobol and Halton low-discrepancy sequences vs. standard Monte Carlo |
| [`multilevel-monte-carlo/`](multilevel-monte-carlo/) | Multilevel Monte Carlo cost reduction for an SDE-driven option price |
| [`importance-sampling-girsanov/`](importance-sampling-girsanov/) | Variance reduction via a Girsanov measure change for deep out-of-the-money pricing |
| [`control-variates/`](control-variates/) | Variance reduction for arithmetic-Asian-option pricing via the closed-form Kemna-Vorst geometric-Asian control |
| [`greeks-pathwise-likelihood-ratio/`](greeks-pathwise-likelihood-ratio/) | Option delta by pathwise vs. likelihood-ratio (score-function) estimators, and why pathwise fails on discontinuous payoffs |
| [`value-at-risk-delta-gamma/`](value-at-risk-delta-gamma/) | Portfolio Value-at-Risk via a delta-gamma approximation, cumulant-generating-function tail inversion, and a control-variate estimator |
| [`portfolio-construction-black-litterman-sdp/`](portfolio-construction-black-litterman-sdp/) | The Black-Litterman Bayesian view-combination model, and the nearest correlation matrix as a semidefinite program via Dykstra-corrected alternating projection |
| [`constrained-optimization/`](constrained-optimization/) | Quadratic programming, a primal-dual interior-point method, and sequential quadratic programming |
| [`first-order-and-quasi-newton-optimization/`](first-order-and-quasi-newton-optimization/) | ADMM (Lasso and distributed consensus), L-BFGS (two-loop recursion vs. full BFGS), and SGD/momentum/Adam on ill-conditioned and disparately-scaled objectives |
| [`sequential-bayesian-filtering/`](sequential-bayesian-filtering/) | The Kalman filter, EKF vs. UKF on a nonlinear tracking problem, the bootstrap particle filter, and HMM forward-backward/Viterbi/Baum-Welch |
| [`financial-instrument-pricing-cpp/`](financial-instrument-pricing-cpp/) | C++: tridiagonal solvers, the Keller Box scheme, exponentially fitted schemes, American options via LCP/PSOR, and a two-factor ADI Asian-option scheme |

## Why these projects

Each notebook turns a mathematical problem into a discrete numerical scheme, and gets its
correctness conditions right: a valid root-finding bracket, a stable time step, a Jacobian that
doesn't vanish, or a scheme's convergence order. Results are checked against something
independent — a closed-form limit, a high-accuracy reference method, a published benchmark, or a
second implementation of the same problem — rather than taken at face value. The later projects
carry this directly into quantitative finance and stochastic simulation: option pricing under
Black-Scholes and Heston, SDE integration schemes measured by their theoretical convergence order,
American-option pricing checked against the original Longstaff-Schwartz benchmark table, Monte Carlo
variance-reduction techniques (quasi-Monte Carlo, multilevel Monte Carlo, importance sampling,
control variates) checked against their theoretical variance-reduction ratios, Greek estimators
checked against closed-form sensitivities, portfolio Value-at-Risk checked against an independent
semi-analytic transform inversion, the Black-Litterman posterior checked against a no-view
consistency identity and a derived (not assumed) invariance property of its own calibration, the
nearest-correlation-matrix solver checked against the Frobenius-optimality guarantee its
alternating-projection correction is supposed to provide, constrained- and large-scale optimization
solvers checked against known optimal points and published benchmarks, sequential filters checked
against their own calibration (innovation consistency) or against each other on problems built to
expose exactly where one's approximating assumptions break down, and the C++ track's
finite-difference and complementarity solvers checked against the same class of closed-form and
published benchmarks used throughout the rest of the repository.

## Notes

- Each notebook is self-contained: open it in Jupyter Notebook/Lab or VS Code and run all cells in
  order. Requires `numpy`, `pandas`, `matplotlib`, `scipy`, `seaborn`, and `sympy`, depending on
  the notebook.
- Cells that prompt for an initial guess, bracket, or matrix/vector entries via `input()` need a
  value typed in when run interactively.
