# Portfolio Construction: Black-Litterman and Semidefinite Programming

Two notebooks on the input side of portfolio construction: replacing an unstable expected-return
vector with a Bayesian posterior that stays anchored to market equilibrium, and repairing a
correlation matrix that has stopped being one.

## Notebooks

| Notebook | Covers |
|---|---|
| [`black_litterman_model.ipynb`](black_litterman_model.ipynb) | Reverse optimization for implied equilibrium returns, Bayesian view combination via Gaussian precision addition, and the resulting portfolio's interpolation between equilibrium and view-driven weights |
| [`nearest_correlation_matrix_sdp.ipynb`](nearest_correlation_matrix_sdp.ipynb) | The nearest correlation matrix as a semidefinite program, solved via Dykstra-corrected alternating projection onto the PSD cone and the unit-diagonal affine set |

## How to build and run

Each notebook is self-contained: open it in Jupyter Notebook/Lab or VS Code and run all cells in
order. Requires `numpy` and `matplotlib`. No portfolio-optimization or SDP library is used for the
core method in either notebook — the mean-variance solve is the same direct linear-algebra
approach used in `constrained-optimization/`, and the SDP is solved by its own projection
algorithm rather than a generic conic solver.

## Black-Litterman model

Derives the implied equilibrium excess returns `Pi = delta * Sigma * w_mkt` from the reverse
mean-variance first-order condition, sets up investor views as a noisy linear system on the
expected-return vector, and derives the Bayesian posterior `mu_BL` by precision-weighted Gaussian
conjugacy — shown explicitly via completing the square on the log-posterior, not quoted. A
no-view consistency check confirms the posterior collapses back to the market weights exactly
(`max abs difference` on the order of machine epsilon). Sweeping the view-confidence multiplier
`c` traces the expected interpolation between `w_mkt` and the view-implied portfolio; sweeping the
prior scale `tau` under the canonical He-Litterman calibration of the view-uncertainty matrix
reveals a subtler and more informative fact, derived analytically and then confirmed numerically
to floating-point precision: `mu_BL` is exactly invariant to `tau` for fixed `c` (the `1/tau`
factor cancels between the prior and view precisions), so `tau` only ever scales the parameter-
uncertainty covariance added on top of the return covariance — it is not, by itself, a second knob
back to equilibrium.

## Nearest correlation matrix via semidefinite programming

States the nearest-correlation-matrix problem in its standard SDP form (minimize the Frobenius
distance to a target matrix subject to positive-semidefiniteness and unit diagonal) and implements
Higham's Dykstra-corrected alternating-projection algorithm: closed-form projection onto the PSD
cone via eigenvalue clipping, closed-form projection onto the unit-diagonal affine set, and a
correction term that restores the memory of the original matrix that plain alternating projection
discards after its first step. Tested on a six-asset correlation matrix built to fail
positive-semidefiniteness the way it actually happens in practice — pairwise estimates from
overlapping but non-identical observation windows, plus manually overridden stress-scenario
entries — the algorithm converges to a matrix that is verified PSD and unit-diagonal to solver
tolerance. Run alongside plain (uncorrected) alternating projection on the same input, the two
feasible points turn out to be numerically close for this instance; the notebook is explicit about
what the Dykstra correction actually guarantees — an unconditional proof of Frobenius-optimality —
rather than overstating the size of the empirical gap it happened to produce here.

## Computational complexity

**Black-Litterman.** For $n$ assets and $k$ views ($n=5$, $k=2$ here), the reverse-optimization
step $\Pi=\delta\Sigma w_{\mathrm{mkt}}$ is a single $O(n^2)$ matrix-vector product. The posterior
$\mu_{BL}$ requires two $n\times n$ inversions ($\tau\Sigma$ and $A+B$, each $O(n^3)$) and one
$k\times k$ inversion of $\Omega$. Because the view-uncertainty matrix $\Omega$ is diagonal by
construction (views are modeled as independent), its inverse is computed once, as an $O(k)$
elementwise reciprocal, and reused for both the $B=P^\top\Omega^{-1}P$ term and the
$P^\top\Omega^{-1}Q$ term — the notebook does this rather than the $O(k^3)$ generic matrix
inversion, computed twice, that the closed-form expression naively suggests. At $n=5$, $k=2$ this
is immaterial in wall-clock terms; the reason it is worth doing correctly is that the sweep over
the confidence multiplier $c$ and the prior scale $\tau$ each call `black_litterman_posterior` 60
times, and at a realistic institutional scale ($n$ in the hundreds, $k$ up to $n$) the $n\times n$
inversions — not $\Omega$ — dominate at $O(n^3)$ per call, so the whole procedure remains cheap
relative to the $O(n^3)$ mean-variance solve that consumes its output.

**Nearest correlation matrix.** Each Dykstra-corrected iteration costs one $O(n^3)$ eigendecomposition
(the PSD projection) plus an $O(n^2)$ diagonal projection and correction update; the algorithm
converges in well under `max_iter=200` iterations for $n=6$, for a total cost of
$O(\text{iterations}\cdot n^3)$. This is the reason the projection algorithm, not a generic
primal-dual SDP solver, is the practitioner's tool for this specific problem: a generic conic
solver would form and factor a Schur-complement system of size $O(n^2)$ at every Newton step,
whereas Dykstra's method needs only an eigendecomposition — asymptotically cheaper per iteration,
and simple enough to implement and audit directly, at the cost of a linear- rather than
quadratic-converging iteration count.
