# Estimating Deltas: Pathwise Derivatives vs. the Likelihood Ratio Method

## Problem

Estimate the sensitivity `∂/∂S(0) E[discounted payoff]` — the option delta — by Monte Carlo, for
three payoffs of increasing structural difficulty under risk-neutral geometric Brownian motion: a
standard European call, a cash-or-nothing digital call, and a path-dependent arithmetic Asian call.

## Method

Two unbiased estimators of `α'(θ) = d/dθ E[Y(θ)]` are derived from first principles, each resting on a
different interchange of limiting operations:

- **Pathwise**: differentiate the payoff itself, `Y'(θ) = lim_{h→0} (Y(θ+h)−Y(θ))/h`, valid whenever
  the difference quotients are uniformly integrable — guaranteed here by writing `Y = f(X(θ))` with
  `f` Lipschitz and `X(θ)` Lipschitz in `θ`. A discontinuous payoff (an indicator function) is **not**
  Lipschitz, so this method fails whenever the payoff has a jump.
- **Likelihood ratio (score function)**: differentiate the *density* of the underlying random vector
  instead of the payoff, `α'(θ) = E[Y · d(log g_θ)/dθ]`. Since probability densities are smooth in
  their parameters even when payoffs are not, this method places no regularity requirement on the
  payoff at all.

The three payoffs are chosen specifically to separate the two methods:

1. **European call** — both methods apply and agree with the closed-form Black–Scholes delta `Φ(d1)`.
2. **Digital call** — the payoff's pathwise derivative is a.s. `0` (the indicator is locally constant
   away from its jump), so the pathwise estimator is *identically zero and uninformative*, while the
   likelihood-ratio estimator remains valid and matches the closed-form digital delta.
3. **Arithmetic Asian call** — no closed form exists. The pathwise estimator follows from the chain
   rule (`S(t_i)` is linear in `S(0)`). The likelihood-ratio estimator is built from the *joint*
   Gaussian density of `(log S(t_1),...,log S(t_m))`, using that `S(0)` enters as a common location
   shift with fixed covariance `Σ_ij = σ²min(t_i,t_j)`; the score for this common-shift parameter is
   `v'(x−μ)` with `v` solving `Σv = 1` (a generalized-least-squares system). Both estimators are
   cross-checked against a central finite-difference estimate built from common random numbers.

## Computational complexity

Both the pathwise and likelihood-ratio estimators are `O(n_paths)`, embarrassingly parallel across
paths, and reuse the *same* simulated terminal (or path) values already needed for the price itself
— a Greek is obtained at essentially the same asymptotic cost as the price, with no extra
simulation. This is the complexity argument for preferring either estimator over a finite-difference
Greek: the notebook's own finite-difference cross-check needs *two* extra full re-simulations
(`S0+h` and `S0-h`) even when reusing the same underlying random numbers, i.e. `3x` the simulation
cost of a single pathwise or LR pass, for an estimator whose bias (from a finite, nonzero `h`) the
pathwise/LR estimators don't have at all. The one non-`O(n)` step in the whole notebook is the
Asian-option likelihood-ratio score, which solves the `m x m` linear system `Sigma v = 1` once via
`np.linalg.solve` (`O(m^3)`, `m=13` monitoring dates here) — negligible next to the
`O(n_paths_asian x m)` cost of simulating the `500,000`-path, `13`-date Asian price itself.

## Files

| File | Contents |
|---|---|
| `greeks_pathwise_likelihood_ratio.ipynb` | Derivations, both estimators, all three payoffs, closed-form/finite-difference cross-checks |

## How to build and run

Open `greeks_pathwise_likelihood_ratio.ipynb` in Jupyter Notebook/Lab or VS Code and run all cells in
order. Requires `numpy`, `scipy`, and `matplotlib`.

## Example output

For `r=0.05`, `σ=0.30`, `S0=50`, `T=0.25`, `K=50`:

| Payoff | Closed form | Pathwise | Likelihood ratio |
|---|---|---|---|
| European call | 0.562903 | 0.563339 | 0.562326 |
| Digital call | 0.052530 | **0.000000 (uninformative)** | 0.052505 |
| Arithmetic Asian call | none | 0.541952 | 0.541043 |

The Asian-option pathwise estimate agrees exactly with a central finite-difference estimate built from
the same underlying random numbers (0.541952 both), and the likelihood-ratio estimate agrees with both
to within its own Monte Carlo standard error. The digital case is the central result: the pathwise
method is not merely less efficient there, it is structurally blind to the option's actual sensitivity.
