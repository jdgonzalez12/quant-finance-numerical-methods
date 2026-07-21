# Higher-Order and Structural Schemes for Stochastic Differential Equations

Six notebooks on numerical schemes for Itô SDEs `dX = a(X,t)dt + b(X,t)dW`, each validated by
measuring its actual convergence order (or, in one case, its numerical stability) against a
reference solution rather than trusting a single sample path. Together they cover the five axes
along which a scheme's theoretical order can fail to appear in practice, or fail to hold at all:
numerical stability at a *fixed* step size, independent of the convergence order itself
(Euler-Maruyama), an under-resolved diffusion term (Milstein/SRK), non-commutative
multi-dimensional noise (the Lévy area), stiffness (the implicit $\theta$-family), and the
strong/weak distinction itself (a scheme judged by pathwise accuracy vs. one judged only by the
accuracy of a distributional functional such as $\mathrm E[X_T]$).

## Notebooks

| Notebook | Covers |
|---|---|
| [`euler_maruyama_method.ipynb`](euler_maruyama_method.ipynb) | The base scheme's well-posedness conditions (Kloeden-Platen Thm. 4.5.3), its derivation from the Itô-Taylor expansion, strong order 0.5 and weak order 1.0 measured on GBM (in separate, differently-tuned experiments, and why that separation is necessary), and its mean-square stability threshold under multiplicative noise |
| [`milstein_method.ipynb`](milstein_method.ipynb) | The Milstein scheme (uses the diffusion coefficient's derivative), applied to geometric Brownian motion and the Cox-Ingersoll-Ross process |
| [`stochastic_runge_kutta.ipynb`](stochastic_runge_kutta.ipynb) | A derivative-free stochastic Runge-Kutta scheme reaching the same strong order without needing `b'`, applied to geometric Brownian motion and a nonlinear SDE |
| [`implicit_theta_scheme.ipynb`](implicit_theta_scheme.ipynb) | The implicit-Euler $\theta$-family for a stiff linear system: A-stability under stiffness vs. strong order (they are independent properties) |
| [`multidimensional_milstein_noncommutative.ipynb`](multidimensional_milstein_noncommutative.ipynb) | The Clark-Cameron example: why multidimensional Milstein needs the Lévy area under non-commutative noise, and collapses to order 0.5 without it |
| [`weak_taylor_order2.ipynb`](weak_taylor_order2.ipynb) | Weak (distributional) vs. strong convergence: the order-2.0 weak Taylor scheme against weak Euler, measured on `E[X_T]` for geometric Brownian motion |

## How to build and run

Each notebook is self-contained: open it in Jupyter Notebook/Lab or VS Code and run all cells in
order. Requires `numpy` and `matplotlib`.

## Euler-Maruyama: well-posedness, both convergence orders, and mean-square stability

States the well-posedness conditions that justify simulating an SDE at all (Kloeden & Platen,
Thm. 4.5.3): global Lipschitz continuity plus linear growth of the drift and diffusion
coefficients give existence and uniqueness of a strong solution with $\sup_{t\le T}\mathbb
E[X_t^2]<\infty$. The scheme itself is then derived directly from the Itô-Taylor expansion —
freezing the drift and diffusion integrands at their left endpoint — making precise exactly which
$O(\Delta t)$ term this discards and the Milstein scheme (`milstein_method.ipynb`) restores.

On geometric Brownian motion, strong order measures $\gamma\approx0.495$ against the theoretical
$0.5$, in the same fixed-path setup used throughout this folder. Weak order is measured
*separately*, deliberately in a lower-diffusion parameter regime with a much larger ensemble
($10^6$ paths): weak error is a bias competing against Monte Carlo sampling noise rather than a
path-wise systematic error, and at the strong-order run's own parameters that noise floor turns
out to swamp the $O(\Delta t)$ signal completely — measured directly and reported, not glossed
over, since it makes a real methodological point about what a "weak order" number requires to be
meaningful. In its own regime, weak order measures $\beta\approx0.970$ against the theoretical
$1.0$: the same discarded Itô-Taylor term is too large to ignore path-wise but has mean zero, so it
costs a full order in the strong criterion while leaving the first moment accurate to a full order
better.

A third property, orthogonal to both convergence orders, is then isolated: **mean-square
stability**. For the linear test equation $dX=\lambda X\,dt+\mu X\,dW$, Itô's formula gives
$\mathbb E[X_t^2]=X_0^2e^{(2\lambda+\mu^2)t}$, so the exact solution decays in mean square iff
$2\lambda+\mu^2<0$. Matching this against the Euler-Maruyama recursion's own conditional second
moment yields an explicit, finite stability threshold $\Delta t^\star = -(2\lambda+\mu^2)/\lambda^2$
on the step size — unlike the exact solution, the explicit scheme only inherits mean-square
stability below $\Delta t^\star$. Simulated directly: at $\Delta t=\tfrac12\Delta t^\star$ the
empirical second moment tracks the exact exponential decay to machine precision; at
$\Delta t=\tfrac32\Delta t^\star$, the identical recursion (same $\lambda,\mu$, same random-seed
family) diverges by six orders of magnitude instead. This is the numerical-stability motivation
behind the fully implicit scheme studied in `implicit_theta_scheme.ipynb`, which is
unconditionally mean-square stable on the same test equation.

## Milstein scheme

Implements `X_{n+1} = X_n + a Δt + b ΔW_n + ½ b b' [(ΔW_n)² − Δt]`.

**Geometric Brownian motion**, `dX = μX dt + σX dW`: strong convergence order is measured against
the exact solution `X_T = X_0 exp((μ−σ²/2)T + σW_T)`, by resolving the same Brownian path at
progressively coarser step sizes (summing fine Wiener increments into coarse ones) and tracking the
mean absolute error at `T` on a log-log plot. Euler-Maruyama and Milstein are run side by side on
identical paths, giving measured orders of essentially 0.5 and 1.0 respectively.

**Cox-Ingersoll-Ross process**, `dX = κ(θ−X)dt + σ√X dW`: since `b(x)b'(x) = σ²/2` is constant for
this diffusion coefficient, the Milstein correction simplifies to `¼σ²[(ΔW_n)² − Δt]` independent of
`X_n`. Parameters are chosen to satisfy the Feller condition `2κθ > σ²`. With no closed-form
path-wise solution available, the scheme is checked against a Milstein simulation on a much finer
grid, and full truncation (`X⁺ = max(X,0)` inside every square root) keeps the discretized path
well-defined.

## Derivative-free stochastic Runge-Kutta

Implements the Platen scheme: a supporting value
`X̄_n = X_n + a Δt + b √Δt`, followed by
`X_{n+1} = X_n + a Δt + b ΔW_n + [b(X̄_n) − b(X_n)] / (2√Δt) · [(ΔW_n)² − Δt]` — reaching the same
strong order 1.0 as Milstein without evaluating `b'` anywhere.

**Geometric Brownian motion**: run on the same paths and parameters as the Milstein notebook, for a
direct, apples-to-apples comparison against Euler-Maruyama.

**A nonlinear SDE**, `dX = −X dt + σ/(1+X²) dW`: `b'(x) = −2σx/(1+x²)²` is not difficult here, but
serves as a worked example of the kind of diffusion coefficient the derivative-free scheme is meant
for. Checked against a fine-grid reference simulation using the same scheme.

## Implicit θ-schemes for stiff systems

Implements the one-parameter implicit-Euler family `Y_{n+1} = Y_n + [θa(t_{n+1},Y_{n+1}) +
(1−θ)a(t_n,Y_n)]Δ + b(t_n,Y_n)ΔW_n` (Kloeden & Platen, Ch. 12.2) on the linear test system `dX = AX
dt + BX dW` with `A = [[−a,−a],[−a,−a]]`, `B = bI`, whose exact solution and Lyapunov spectrum
(`λ₁=0, λ₂=−2a` for the deterministic part) are both closed-form.

**Stability under stiffness** (`a=25, b=2`, giving fast-mode decay rate `−52`): at a step size that
lies just inside the explicit scheme's (`θ=0`) instability region, the explicit trajectory blows up
by two orders of magnitude while the fully implicit scheme (`θ=1`), run on the identical driving
path, tracks the bounded exact solution.

**Strong order** (Kloeden & Platen, PC-Exercise 12.2.3, `a=5, b=0.01` and a second, better-scaled
parameter set): every member of the θ-family — explicit, trapezoidal, and fully implicit alike —
measures strong order `γ≈0.5`, confirming that `θ` is a stability control, not an accuracy control.
The literal book parameters are reproduced first and shown to *not* yet exhibit a clean power law at
the tested step sizes (a drift/diffusion crossover effect, explained rather than hidden); a
second, diffusion-dominated parameter regime isolates the asymptotic order cleanly.

## Multidimensional Milstein under non-commutative noise

The Clark & Cameron (1980) example: `X¹=W¹`, `dX²=X¹dW²`, a case where the two noise directions do
not commute. The Milstein scheme's correction term for `X²` is the double Itô integral `I_(1,2)`,
which decomposes into a symmetric part `½ΔW¹ΔW²` and the antisymmetric **Lévy area**. The naive
scheme most scalar-case intuition suggests — dropping the Lévy area — measures strong order `≈0.5`,
no better than Euler-Maruyama, because the discarded term is itself `O(Δ)` in scale, not a
higher-order correction. Restoring an approximation of the true `I_(1,2)` (via a resolution strictly
between the coarse step and the reference path's finest resolution, scaled to avoid both a
tautological reconstruction and an under-resolved one — see the notebook's Method section for why
the scaling law matters) recovers the theoretical order `γ≈1.0`.

## Weak vs. strong convergence: the order-2.0 weak Taylor scheme

For geometric Brownian motion, `E[X_T] = x0·e^{aT}` in closed form, independent of `b`. Weak Euler
(algebraically identical to strong Euler-Maruyama, only the error criterion differs) measures weak
order `β≈1.0` against this target, while the order-2.0 weak Taylor scheme — which adds the `bb'`
Milstein-type correction plus two further Itô-Taylor terms, one of them driven by the exactly
jointly-Gaussian time-space Lévy area `ΔZ_n = ∫(W_s−W_{t_n})ds` — measures `β≈2.0`. Unlike the
multidimensional Lévy area elsewhere in this folder, `ΔZ_n` requires no separate approximation
scheme: its exact conditional law given `ΔW_n` is derived directly from `Cov(ΔW_n,ΔZ_n)=½Δ²` and
`Var(ΔZ_n)=⅓Δ³`, and sampled exactly.

## Computational complexity and efficiency

Every path-simulation scheme in this folder (Euler-Maruyama, Milstein, SRK, the implicit
$\theta$-family, the naive/corrected Clark-Cameron schemes, weak Euler/Taylor2) costs `O(n_steps)`
per path and is fully vectorized across paths with NumPy array operations, so the total cost is
`O(n_paths * n_steps)` with no Python-level loop over paths anywhere — only over time steps, which
is unavoidable: each scheme is a genuine recursion, $X_{n+1}$ a function of $X_n$, not an
embarrassingly parallel map over the time axis the way it is over the path axis. This is why every
notebook's inner loop is `for i in range(n_steps): ... X = X + ...` operating on a full
`(n_paths,)`-shaped array per iteration, rather than a loop over paths.

- `euler_maruyama_method.ipynb` / `milstein_method.ipynb` / `stochastic_runge_kutta.ipynb`: `O(1)`
  work per step per path (a handful of elementwise array operations), so `O(n_paths * n_steps)`
  total — `20{,}000 * 4096` for the GBM strong-order runs, `10^6 * 16` (finest tested $\Delta$) for
  the weak-order run, which is why the weak-order experiment needs a far larger `n_paths`: its
  signal is a bias competing against Monte Carlo noise of order $1/\sqrt{n_{\text{paths}}}$, not a
  per-path pathwise error, so beating the noise floor costs orders of magnitude more paths at
  comparable $\Delta t$.
- `implicit_theta_scheme.ipynb`: the "implicit" solve is a *fixed*, `2x2` matrix inverse
  (`lhs_inv`) computed once outside the time loop (since $A,B$ are time-invariant), then applied
  as one matrix-vector product per step — `O(1)` per step exactly as in the explicit case, so
  implicitness buys unconditional stability here at zero asymptotic cost, precisely because the
  test system is linear and low-dimensional. A nonlinear drift would instead require a per-step
  Newton solve, trading `O(1)` steps for a small constant number of Jacobian solves each.
- `multidimensional_milstein_noncommutative.ipynb`: the naive scheme is `O(n_steps)` per path,
  same as scalar Milstein. The *corrected* scheme's Lévy-area reconstruction aggregates
  `K_aux = n_steps` medium sub-increments per coarse block to avoid the tautology described above,
  which costs `O(n_steps)` per block and hence **`O(n_steps^2)` per path overall** — asymptotically
  more expensive than every other scheme in this folder, and worth stating plainly rather than
  glossing over: this notebook's Lévy-area construction is chosen for conceptual transparency (it
  is literally a Riemann-sum approximation of the defining double integral), not for production
  efficiency. Kloeden & Platen's own truncated Karhunen-Loève series approximates the same
  quantity with a small, fixed number of terms per step, `O(n_steps)` overall — the standard choice
  in a production SDE integrator for non-commutative noise, traded here for the simpler
  construction's transparency.
- `weak_taylor_order2.ipynb`: both schemes are `O(1)` work per step per path (weak Taylor2 draws
  one extra Gaussian per step for $\Delta\widetilde W_n$ and evaluates a few more terms), so both
  remain `O(n_paths * n_steps)` — the extra accuracy order is bought with a larger constant factor
  per step, not a higher-complexity algorithm, which is precisely why a weak-order-2 scheme is
  preferable to naively shrinking $\Delta t$ under weak Euler whenever the constant-factor cost of
  the extra terms is cheaper than the $O((\Delta t_{\text{old}}/\Delta t_{\text{new}})^2)$ blow-up
  in step count a matching accuracy gain would otherwise require.
