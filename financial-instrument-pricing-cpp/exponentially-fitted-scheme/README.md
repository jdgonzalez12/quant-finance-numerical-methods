# Exponentially Fitted Schemes for Convection-Dominated Problems

## Problem

Black-Scholes is a special case of a convection-diffusion-reaction equation, and whenever the
convection coefficient dominates the diffusion coefficient at the mesh scale — low volatility,
high rate, or simply a coarse grid near large `S` where the diffusion coefficient `(1/2)sigma^2 S^2`
grows more slowly than the convection coefficient `rS` relative to it — a standard centered-difference
discretization develops spurious, non-physical oscillations. This is not a stability failure in the
von Neumann sense (the scheme can remain linearly stable while still oscillating); it is a failure of
the *discrete maximum principle*: the difference operator stops being monotone, so it no longer maps
non-negative data to non-negative solutions, and the truncation error's leading term is no longer
controlled by a mesh-independent constant. Il'in (1969) and, for the Black-Scholes setting
specifically, Duffy (1980) showed how to repair this without sacrificing accuracy.

## Method

**Motivating case.** Consider `sigma u'' + mu u' = 0` on `(A,B)` with `u(A)=beta_0`, `u(B)=beta_1`
and `sigma, mu > 0` constant. The centered scheme

```
sigma D+D- U_j + mu D0 U_j = 0
```

(`D+D-` the standard second difference, `D0` the centered first difference) is consistent to `O(h^2)`,
but its coefficient matrix is diagonally dominant only when the **cell Peclet number**
`Pe_h := mu h / (2 sigma)` satisfies `Pe_h <= 1`; once `Pe_h > 1` the discrete maximum principle is
violated and the computed solution oscillates around the true, monotone one, with an amplitude that
does not shrink as `h -> 0` at fixed `Pe_h` — refining the mesh alone does not fix it unless `h` is
refined fast enough to bring `Pe_h` back under 1.

**The fitting factor.** Il'in's repair multiplies the second-difference term by a factor

```
rho = Pe_h coth(Pe_h)
```

chosen so that the discrete equation `sigma rho D+D- U_j + mu D0 U_j = 0` reproduces the *exact*
solution of the constant-coefficient problem at every mesh point, for every `h` — not merely to
higher order, but exactly, since `rho` is derived by matching the difference equation's characteristic
roots to the true ODE's exponential solution `u(x) = beta_0 + (beta_1-beta_0)(1-e^{-mu(x-A)/sigma})/(1-e^{-mu(B-A)/sigma})`
rather than by a Taylor-series truncation argument. As `Pe_h -> 0`, `rho -> 1` and the fitted scheme
degrades gracefully to the ordinary centered scheme — the fitting is inert exactly where it is not
needed.

**Variable coefficients and monotonicity.** For the general two-point problem
`sigma(x) u'' + mu(x) u' + b(x) u = f(x)` with `sigma >= 0`, `mu >= alpha > 0`, `b <= 0`, the fitted
scheme uses `rho_j = (mu_j h / 2 sigma_j) coth(mu_j h / 2 sigma_j)` at each node, and Il'in (1969)
proves the resulting tridiagonal matrix has strictly positive off-diagonal entries and a negative
diagonal — i.e. it is monotone (an M-matrix) — *for every* `sigma, mu, h > 0`, unlike the unfitted
scheme whose monotonicity is conditional on `Pe_h <= 1`. The consequence is a uniform stability bound
`|U_j| <= |beta_0| + |beta_1| + alpha^{-1} max_k |f_k|` and a convergence bound `|u(x_j) - U_j| <= Mh`
with `M` independent of both `h` and `sigma` — the fitted scheme remains first-order accurate and
oscillation-free uniformly across the entire convection-dominated-to-diffusion-dominated spectrum,
whereas the unfitted scheme's error constant blows up as `sigma -> 0` at fixed `h`.

**Application to Black-Scholes.** Extending the same fitted operator to the parabolic problem with a
fully implicit (or, as tested here, Crank-Nicolson) discretization in `tau` gives a scheme whose
spatial part inherits the monotonicity above at every time level. The effect the book documents is
specifically that the *price* itself may show only a mild discrepancy under the unfitted scheme, but
the *delta* — the quantity actually used for hedging — develops a much more visible oscillation, since
the payoff's discontinuous second derivative at the strike is what excites the effect, and
differentiating amplifies high-frequency error.

## Computational complexity

Both the centered and fitted schemes reduce to a Thomas (tridiagonal) solve per time step, `O(J)` in
time and memory (see `tridiagonal-solvers/`), so `N` Crank-Nicolson steps on `J` spatial cells cost
`O(NJ)` total — computing the fitting factor `rho_j` itself is `O(1)` per node, an `O(J)` addition to
an already-`O(J)` step, not a new complexity class. Part B needs a value at an intermediate step
(`reportAfterSteps`) and at the final step `N`, for both the centered and fitted operator; rather than
running the time march twice per operator (once to the intermediate step, discarded, and again to
`N`), `runScheme` marches once to `N` and snapshots the grid in passing, halving the number of Thomas
solves this program performs. Per-step scratch buffers (the tridiagonal coefficient arrays and the
solver's output) are allocated once outside the time loop and reused every step, rather than
reallocated `N` times.

## Files

| File | Contents |
|---|---|
| `exponentially_fitted_scheme.cpp` | Part A: exact reproduction of the constant-coefficient ODE result. Part B: fitted vs. centered Crank-Nicolson applied to Black-Scholes in a convection-dominated regime |

## How to build and run

```
g++ -O2 -std=c++17 -Wall -o exponentially_fitted_scheme exponentially_fitted_scheme.cpp
./exponentially_fitted_scheme
```

No external dependencies beyond the C++ standard library.

## Example output

**Part A** (`sigma=0.01`, `mu=1`, `J=20`, cell Peclet number `2.5`): the centered scheme's error
reaches `0.435` at the node nearest the boundary layer and its consecutive differences change sign
19 times (a textbook oscillatory solution); the fitted scheme matches the exact solution to
floating-point round-off at every node, with zero sign changes.

**Part B** (Black-Scholes, `sigma=0.01`, `r=0.50`, `K=25`, `S_max=100`, `T=0.25`, `J=120`, `N=50`,
Crank-Nicolson in time): the price itself does not visibly oscillate under either scheme at this
resolution, but the discrete delta does — the centered scheme's delta swings between roughly `0.86`
and `1.23` around the strike (11 sign changes in its consecutive differences), while the fitted
scheme's delta rises monotonically and saturates at exactly `1.0` (a single sign change, at the
floating-point noise floor) — precisely the "more pronounced in the delta" failure mode the method
was designed to remove.
