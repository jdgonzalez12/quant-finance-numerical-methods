# Finite Differences, Numerical Integration, and ODE Solvers

Two notebooks covering finite-difference differentiation, numerical integration (rectangles,
trapezoids, Simpson's 1/3, Monte Carlo), and initial-value ODE solvers (Euler, midpoint, Heun),
each checked against a high-accuracy reference.

## Notebooks

| Notebook | Covers |
|---|---|
| [`finite_differences_integration_heun.ipynb`](finite_differences_integration_heun.ipynb) | Finite-difference derivatives of increasing order, velocity/acceleration from discrete data, three integration schemes, and Heun's method for an ODE |
| [`finite_differences_integration_odes.ipynb`](finite_differences_integration_odes.ipynb) | Finite-difference derivatives against `scipy.optimize.approx_fprime`, three integration schemes at several resolutions, and a first- and second-order ODE solved with Euler, midpoint, and Heun's method |

## How to run

Each notebook is self-contained: open it in Jupyter Notebook/Lab or VS Code and run all cells in
order. Requires `numpy`, `pandas`, `matplotlib`, and `scipy`.

## Computational complexity

- **Finite-difference derivatives** are `O(1)` per evaluation point regardless of order (a fixed
  stencil of 2–4 function evaluations), with truncation error `O(h)` for a first-order
  forward/backward difference and `O(h^2)` for the centered and higher-order formulas used here —
  the accuracy-vs-`h` tradeoff the notebooks sweep over, not a cost tradeoff.
- **Numerical integration** over `n` subintervals costs `O(n)` function evaluations for the
  rectangle rule, trapezoids, and Simpson's 1/3 alike; what differs is the truncation error per
  unit of that cost: `O(h)` for rectangles, `O(h^2)` for trapezoids, `O(h^4)` for Simpson's 1/3
  (`h=(b-a)/n`) — so Simpson's rule reaches a target accuracy at a far smaller `n`, which is exactly
  why it is used as the higher-accuracy reference against which the coarser rectangle pass is
  checked.
- **ODE solvers** (Euler, midpoint, Heun) are inherently sequential in time — each step's state
  depends on the previous one, so there is no `n_steps` axis to vectorize away — costing `O(n)`
  for `n = (t_{\text{end}}-t_0)/\Delta t` steps, each `O(1)` for the scalar and 2-state systems
  solved here. Global error is `O(\Delta t)` for explicit Euler and `O(\Delta t^2)` for the
  midpoint and Heun (explicit predictor-corrector) schemes, consistent with Heun's method costing
  exactly two function evaluations per step against Euler's one, in exchange for a full order of
  accuracy.

## Finite differences, integration, and Heun's method

**1. Finite-difference derivative approximations of increasing order.** First-, second-, and
third-order forward-difference formulas for `f(x) = x*cos(x)*sin(sqrt(1+x²)/2)` at `x = π`,
compared at several step sizes `h` against a high-accuracy reference (second-order formula at
`h = 0.0001`).

**2. Velocity and acceleration from discrete position data.** Given six evenly-spaced position
samples, velocity and acceleration are recovered via forward/backward differences at the
endpoints and centered differences in the interior.

**3. Numerical integration: rectangles, Simpson's 1/3, and Monte Carlo.** All three methods
applied to `f(x) = sin(sin(x))` on `[0, π]`, checked against `scipy.integrate.quad`; a coarse
rectangle-rule pass is also checked against a finer Simpson reference, and all three methods are
compared directly against each other.

**4. Heun's method (explicit predictor-corrector) for an ODE.** Solves
`dy/dx = yx² - y`, `y(0) = 1`, over `x ∈ [0, 3]` with step `h = 0.2`.

## Numerical differentiation and integration

**I. Finite-difference derivatives.** For three different functions, first- and third-order
finite-difference derivatives (step `h = 0.01`) are computed at one or two evaluation points and
compared, in absolute and relative terms, against a reference derivative from
`scipy.optimize.approx_fprime`.

**II. Numerical integration.** For three different integrands, the area under the curve is
computed via left-Riemann sums, trapezoids, and Simpson's 1/3 rule at `n = 2, 4, 10`, and compared
against `scipy.integrate.quad` in absolute and relative error.

**III. ODEs via Euler, midpoint, and Heun's method.** A first-order ODE
(`dx/dt = -x*sin²(t)`, `x(0) = 1`) and a second-order ODE (`y'' = t - y`, `y(0) = 2`, `y'(0) = 1`,
reduced to a first-order system) are both solved over `t ∈ [0, 3]` with step `dt = 0.1`, using all
three methods, and the resulting trajectories are compared graphically.
