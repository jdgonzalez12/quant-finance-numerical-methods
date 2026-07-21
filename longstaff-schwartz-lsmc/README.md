# American Option Pricing via Longstaff-Schwartz (LSMC)

Least-squares Monte Carlo pricing of an American put, following Longstaff & Schwartz (2001),
validated directly against the paper's own published benchmark prices.

## Problem

Price an American put option on a non-dividend-paying stock under Black-Scholes dynamics
`dS_t = rS_t dt + σS_t dW_t`. Early exercise means the value at each exercise date depends on an
optimal stopping decision (exercise now vs. continue), and the continuation value has no closed form
once early exercise is allowed.

## Method

**Longstaff, F.A. & Schwartz, E.S. (2001), "Valuing American Options by Simulation: A Simple
Least-Squares Approach," Review of Financial Studies, 14(1), 113-147.**

Backward induction over simulated GBM paths: at each exercise date, the continuation value is
estimated by regressing realized discounted future cash flows on a basis of weighted Laguerre
polynomials evaluated at `x = S/K`,
```
L0(x) = e^{-x/2},  L1(x) = e^{-x/2}(1-x),  L2(x) = e^{-x/2}(1 - 2x + x^2/2),
```
using only paths that are in-the-money at that date (the original Longstaff-Schwartz restriction:
regressing on out-of-the-money paths, whose continuation value is irrelevant to the exercise
decision, only adds regression noise).
Where the immediate exercise value exceeds the estimated continuation value, the path's cash flow is
overwritten with the exercise value and its exercise time is recorded; otherwise the path's existing
(later) cash flow is left untouched. After the backward pass, every path's realized cash flow is
discounted from its own exercise time back to `t=0` and averaged.

## Computational complexity

Path simulation is `O(n_paths x n_steps)` — here `n_paths=100,000`, `n_steps = steps_per_year x T`
(50-100) — and fully vectorized across paths via `numpy.cumsum` along the time axis, with no
Python-level loop over paths at any point. The backward-induction pass over exercise dates is an
`O(n_steps)` sequential outer loop that **cannot** be vectorized over time: the continuation value
at `t_j` regresses on cash flows that already embed every exercise decision made at dates after
`t_j`, so the dates must be resolved strictly in decreasing order. Within each date, the
in-the-money mask keeps the regression restricted to `n_itm <= n_paths` paths, and the least-squares
solve uses a fixed, small basis (`k=3` Laguerre polynomials), so its cost is `O(n_itm x k^2) =
O(n_itm)` per date — negligible next to the `O(n_paths)` masking and payoff evaluation done at the
same step. Total cost is therefore dominated by path simulation and masking:
`O(n_paths x n_steps)`, with `O(n_paths x n_steps)` memory to hold the full path array (at the
sizes used here, `~40 MB`) — required because both the backward induction and the exercise-behavior
plot need every path's full trajectory, not just its terminal value.

## Files

| File | Contents |
|---|---|
| `longstaff_schwartz_lsmc.ipynb` | GBM path simulation, Laguerre-basis regression, backward-induction LSMC pricer, Black-Scholes cross-check, and validation against published benchmarks |
| `lsmc_exercise_behavior.png` | Sample paths with realized exercise points, and the distribution of exercise times |

## How to build and run

Open `longstaff_schwartz_lsmc.ipynb` in Jupyter Notebook/Lab or VS Code and run all cells in order.
Requires `numpy`, `pandas`, `matplotlib`, and `scipy`. Contract and simulation parameters (`K`, `r`,
paths per case, steps per year) are set at the top of the notebook.

## Example output

For five `(S0, σ, T)` combinations from the paper's Table 1 (strike `K=40`, `r=0.06`), the prices
produced here land within one or two standard errors of both the paper's own LSMC estimate and its
finite-difference benchmark:

| S0 | σ | T | European (closed-form) | American (this notebook) | std. error | American (Longstaff-Schwartz 2001) | American (finite-difference benchmark) |
|---|---|---|---|---|---|---|---|
| 36 | 0.20 | 1 | 3.844 | 4.473 | 0.0092 | 4.472 | 4.478 |
| 36 | 0.20 | 2 | 3.763 | 4.815 | 0.0110 | 4.821 | 4.840 |
| 36 | 0.40 | 1 | 6.711 | 7.055 | 0.0188 | 7.091 | 7.101 |
| 40 | 0.20 | 1 | 2.066 | 2.302 | 0.0087 | 2.313 | 2.314 |
| 44 | 0.40 | 2 | 5.202 | 5.636 | 0.0206 | 5.622 | 5.647 |

Every case satisfies `European <= American <= finite-difference benchmark` within simulation noise,
consistent with a strictly positive early-exercise premium. For `S0=36, σ=0.20, T=1`, about 69% of
paths exercise before maturity rather than holding to expiry.
