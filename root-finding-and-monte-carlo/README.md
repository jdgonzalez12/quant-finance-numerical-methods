# Root-Finding and Monte Carlo Integration

Six notebooks covering the standard open and closed methods for single- and multi-variable
root-finding, plus Monte Carlo estimation of definite integrals. Each case compares methods by
convergence behavior (iterations needed vs. target error) rather than reporting a single run.

Every root-finder and the Monte Carlo estimator itself are implemented directly from their
defining iteration or sampling scheme, with no `scipy.optimize` root-finder and no
`scipy.integrate` quadrature rule standing in for the algorithm under study; `scipy.optimize.brentq`
and `scipy.integrate.quad` appear exactly twice in this folder, and only as an independent
reference value to check a from-scratch result against, never as the method being implemented.

## Notebooks

| Notebook | Covers |
|---|---|
| [`root_finding_methods_survey.ipynb`](root_finding_methods_survey.ipynb) | Bisection vs. false position, Newton vs. false position, fixed-point vs. Newton, and 2D Newton-Raphson |
| [`fixed_point_vs_newton.ipynb`](fixed_point_vs_newton.ipynb) | Four equations, each solved by fixed-point iteration and Newton's method |
| [`nonlinear_systems_newton.ipynb`](nonlinear_systems_newton.ipynb) | Multivariate Newton-Raphson for three nonlinear systems (2x2, 2x2, 3x3) |
| [`bisection_false_position.ipynb`](bisection_false_position.ipynb) | Bisection, false position, and a bisection/false-position hybrid, on four equations |
| [`applied_root_finding_problems.ipynb`](applied_root_finding_problems.ipynb) | Three applied engineering problems reduced to root-finding |
| [`monte_carlo_integration.ipynb`](monte_carlo_integration.ipynb) | Monte Carlo estimation of four definite integrals, checked against `scipy.integrate.quad` |

## How to run

Each notebook is self-contained: open it in Jupyter Notebook/Lab or VS Code and run all cells in
order. Requires `numpy`, `scipy`, `matplotlib`, `seaborn`, and `sympy`. Cells that prompt for an
initial guess or bracket via `input()` need a value typed in when run interactively.

## Computational complexity

Every method here is scalar or low-dimensional, so the interesting cost is in the *iteration
count*, not the per-iteration work (each iteration is `O(1)` arithmetic, or `O(k^3)` for a `k x k`
Jacobian solve in the multivariate cases below, negligible at `k=2,3`).

- **Bisection**: the bracket width halves every iteration, so reaching absolute error `tol` from
  an initial bracket `[a,b]` takes exactly `⌈log2((b-a)/tol)⌉` iterations — a guaranteed bound,
  independent of `f`, that follows directly from the bracket-halving invariant.
- **False position**: also guarantees convergence from a bracket, but the secant construction can
  let one endpoint stall for many iterations when `f` is strongly convex/concave near the root, so
  the iteration count has no `f`-independent closed form the way bisection's does; the hybrid
  scheme in `bisection_false_position.ipynb` reverts to a bisection step whenever the secant slope
  exceeds a threshold specifically to bound this worst case.
- **Newton's method**: once the iterate is close enough to a simple root (`f'` bounded away from
  `0` nearby), convergence is quadratic, `|x_{n+1}-x^*| \le C|x_n-x^*|^2`, so the number of
  iterations to reach `tol` scales like `O(log log(1/tol))` — doubling the number of correct
  digits every step — versus fixed-point iteration's linear rate `|x_{n+1}-x^*|\approx
  |g'(x^*)|\,|x_n-x^*|`, which needs `O(log(1/tol)/\log(1/|g'(x^*)|))` iterations, growing only
  linearly in the number of correct digits required.
- **Monte Carlo integration**: `N` independent samples cost `O(N)` total, and the estimator's
  standard error shrinks as `O(1/\sqrt N)` by the central limit theorem — *dimension-independent*,
  which is exactly why the notebook evaluates it at `N=10,100,1000,10000` (each 10x increase in
  cost buys only a `~3.16x` reduction in error) rather than treating it as competitive with
  Simpson's rule in one dimension; its advantage only appears in higher-dimensional integrals,
  where deterministic quadrature's cost grows exponentially in dimension and Monte Carlo's does not.

## Root-finding methods survey

Four independent problems, each solved with two methods and compared by convergence behavior
against the analytic iteration estimate `log2((b-a)/tol)` where applicable.

**1. Bisection vs. false position** for `sin(exp(-3x^2+2)) - 2x^3 = 0` on `[0, 2]`.

**2. Newton vs. false position** for a highly oscillatory function on `[-1, 0]`:
`sin(7x)e^(-x/9) - 4cos(5x)e^(x/5) - 12sin(7x)e^(-x/7) + x^3/21 - 5 = 0`.

**3. Fixed-point iteration vs. Newton's method** for `(e^-x - 0.5)sin(x) - x + 5 = 0`, using the
direct choice `g(x) = f(x) + x`.

**4. Newton-Raphson for a 2D nonlinear system**: `sin(e^(x^2-y^2)) - 3y = 0` and
`cos(cos(xy^3)) - 3x^2y = 0`, solved with a numerically approximated Jacobian.

## Fixed-point iteration vs. Newton's method

Four independent equations, each solved with both open methods. For fixed-point iteration, when
the direct choice `g(x) = f(x) + x` does not converge, an alternative `g(x)` is derived
analytically so that it does. Each root is also shown graphically, and both methods report
iterations needed across a sweep of target errors from `1e-2` down to `1e-12`.

- a) `x^2 - 4x + 4 - ln(x) = 0`
- b) `x + 1 - 2sin(pi*x) = 0`
- c) `1/2 + x^2/4 - x*sin(x) - cos(2x)/2 = 0`
- d) `1564 = 1000e^x + 435(e^x - 1)/x`

## Newton-Raphson for nonlinear systems

Three systems of increasing dimension, each solved with the multivariate Newton-Raphson method
using a numerically approximated Jacobian.

- **A.** Intersection of two circles: `(x-4)^2 + (y-4)^2 = 4` and `x^2 + y^2 = 16`.
- **B.** `y + x^2 - 0.5 = 4` and `y = x^2 - 5xy`.
- **C.** A 3x3 system: `3x - cos(yz) - 1/2 = 0`, `4x^2 - 625y^2 + 2y - 1 = 0`,
  `e^(-xy) + 20z + (10*pi - 3)/3 = 0`.

## Bisection, false position, and a hybrid method

Four independent equations, each solved with bisection, false position, and a hybrid scheme that
switches to bisection whenever the false-position secant slope gets too steep (`|Δf/Δx| > 3`).
Each case reports iterations vs. target error from `1e-2` down to `1e-12`.

- a) `sqrt(x) - cos(x) = 0`
- b) `x - 2^-x = 0`
- c) `2x*cos(x) - (x+1)^2 = 0`
- d) `e^x - 2 = cos(e^x - 2)`

## Applied root-finding problems

Three applied engineering problems, each reduced to a single-variable root-finding equation and
solved with an open method.

**A. Channel depth from Manning's equation.** Solve
`Q = (1/n)*(BH)^(5/3)*S^(1/2) / (B+2H)^(2/3)` for the channel height `H` required to drain
50 m³/s over a 35 m-wide channel with roughness 0.045 and slope not exceeding 0.1°, via Newton's
method.

**B. Equilibrium angle of a loaded bar.** A bar under a 200 lb weight, a 100 lb-ft couple moment,
and a spring (unstretched length 2 ft, `k = 50` lb/ft) is analyzed down to
`1800*sin(θ)cos(θ) - 100 - 600cos(θ) = 0`, solved with Newton's method.

**C. Launch angle for a projectile with a fixed range.** Given an initial speed of 20 m/s, a
40 m horizontal distance, and a 1.8 m launch height, the trajectory equation reduces to
`40cos²(θ)tan(θ) + 1.8cos²(θ) - 19.62 = 0`. Solved first by fixed-point iteration (with `g(x)`
derived by isolating `tan(x)` in `f(x) = 0`), then cross-checked with `scipy.optimize.brentq`.

## Monte Carlo integration

Four definite integrals estimated by Monte Carlo sampling (rejection method), compared against a
high-accuracy reference (`scipy.integrate.quad`). For each case, percent error is computed at
`N = 10, 100, 1000, 10000` random draws, and `N` vs. error is plotted to show convergence.

- a) `∫₁^1.5 x²ln(x) dx`
- b) `∫₀^5 [2x*cos(2x) - (x-2)²] dx`
- c) `∫₀^48 sqrt(1 + cos³(x)) dx`
- d) `∫_e^2e 1/(x*ln(x)) dx`
