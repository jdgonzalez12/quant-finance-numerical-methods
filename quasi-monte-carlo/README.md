# Quasi-Monte Carlo Option Pricing: Sobol and Halton Sequences

## Problem

Price a European call option under Black-Scholes by simulating the terminal underlying price
`S_T = S_0 exp((r - σ²/2)T + σ√T Z)` with `Z ~ N(0,1)`, and compare the convergence rate of the
pricing error against the closed-form Black-Scholes value across three sampling schemes: standard
(pseudo-random) Monte Carlo, and quasi-Monte Carlo with Sobol and Halton low-discrepancy sequences.

## Method

Standard Monte Carlo draws i.i.d. normals via the inverse-CDF transform of pseudo-random uniforms;
its RMSE decays as `O(N^-1/2)` regardless of how those uniforms are generated, by the classical
CLT argument alone — it does not depend on smoothness of the integrand. Quasi-Monte Carlo replaces
the pseudo-random uniforms with a **low-discrepancy sequence**: a deterministic (or scrambled)
point set `{u_1,...,u_N} in [0,1]^d` whose star discrepancy
`D*_N = sup_{B} | (#{i : u_i in B} / N) - vol(B) |` (the sup ranging over all axis-aligned boxes
`B = prod_j [0,b_j) subset [0,1]^d`) decays as `O((log N)^d / N)` rather than the `O(N^-1/2)` of a
random point set — a measure of how evenly the points fill the hypercube in the worst case over
all such boxes, not merely on average. By the Koksma-Hlawka inequality, the integration error is
bounded by `D*_N` times the integrand's total variation (in the Hardy-Krause sense), which is why
low discrepancy translates into faster convergence *only* when the integrand is smooth enough for
that variation to be finite and not too large — while keeping the same inverse-CDF transform. Two constructions are used, both scrambled to allow randomized-QMC error
estimation across independent runs: a **Sobol sequence** (base-2 digital net,
`scipy.stats.qmc.Sobol`) and a **Halton sequence** (van der Corput sequences in coprime bases,
`scipy.stats.qmc.Halton`). For a smooth integrand of low effective dimension, QMC error decays
close to `O(N^-1)` — a full order faster than standard Monte Carlo.

A second, higher-dimensional test prices a discretely-monitored arithmetic Asian call, averaging a
GBM path over `d=16` equally-spaced monitoring dates, with the Sobol/Halton sequences generated
directly in `d` dimensions. This is a more realistic setting for the QMC/MC comparison, since a
single effective dimension (as in the plain European call) is the case most favorable to
low-discrepancy sequences.

## Computational complexity

Generating `n` points costs `O(n)` for Halton (van der Corput digit-reversal in each of `d`
coprime bases) and `O(n log n)` for Sobol (digital-net construction), both negligible next to the
`O(n)` payoff evaluation that follows; the real quantity of interest is not per-sample cost but
**samples required for a target accuracy**, since QMC's cost per sample is the same order as
standard Monte Carlo's. A standard-MC estimator needs `O(epsilon^-2)` samples to reach RMSE
`epsilon` (the `N^-1/2` rate measured here); a QMC estimator on a smooth, effectively
low-dimensional integrand needs closer to `O(epsilon^-1)` samples (the near-`N^-1` rate measured
on the 1-D European call) — a **quadratic** reduction in the sample count needed for the same
accuracy, not merely a constant-factor speedup. This is exactly why the gain collapses on the
16-dimensional Asian call: the *effective dimension* of an integrand (in the
Caflisch-Morokoff-Owen truncation/superposition sense — the number of coordinates, or low-order
coordinate interactions, that actually carry the integrand's variance) governs QMC's rate, not the
nominal dimension `d` used to generate the sequence; an equally-weighted 16-date path average has
effective dimension close to its nominal one under a plain coordinate-wise Sobol/Halton mapping, so
the quadratic-in-samples advantage is largely lost, exactly as measured (`N^-0.49`, `N^-0.70` vs.
the `N^-1.09`, `N^-0.97` seen in the 1-D case).

## Files

| File | Contents |
|---|---|
| `quasi_monte_carlo.ipynb` | Both pricing experiments, convergence-rate estimation, and plots |
| `qmc_1d_convergence.png` | RMSE vs. sample size, European call (log-log, with `N^-1/2`/`N^-1` reference slopes) |
| `qmc_asian_convergence.png` | RMSE vs. sample size, 16-dimensional Asian call |

## How to build and run

Open `quasi_monte_carlo.ipynb` in Jupyter Notebook/Lab or VS Code and run all cells in order.
Requires `numpy`, `matplotlib`, and `scipy`.

## Example output

For a European call (`S_0=100`, `K=100`, `r=0.03`, `σ=0.20`, `T=1`, closed-form price `9.4134`),
the empirical RMSE decay rates measured across `N = 2^6` to `2^16` are `N^-0.48` (standard MC,
matching the theoretical `-1/2`), `N^-1.09` (Sobol), and `N^-0.97` (Halton) — both low-discrepancy
sequences decay close to a full order faster than standard Monte Carlo, as expected for a smooth,
effectively one-dimensional integrand.

For the 16-dimensional Asian call (reference price `5.5505` from a 4-million-path standard-MC
run), the advantage is much smaller: standard MC decays at `N^-0.48` as before, but Sobol only
reaches `N^-0.49` — essentially no improvement — while Halton reaches `N^-0.70`. This is a real
and expected effect, not a bug: the QMC advantage depends on the integrand having low *effective*
dimension, and a plain dimension-by-dimension mapping of a 16-dimensional low-discrepancy sequence
onto an equally-weighted path average does not concentrate the sequence's uniformity where it
matters most. In practice this gap is closed with a dimension-reduction path construction (e.g. a
Brownian bridge or principal-component construction of the path), which was intentionally left out
here to keep the comparison isolated to the sampling sequence itself.
