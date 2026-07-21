# First-Order and Quasi-Newton Optimization for Large-Scale Objectives

Three notebooks on unconstrained optimization at a scale where forming a Hessian, or even
minimizing a coupled objective jointly, is not an option: splitting methods, memory-limited
quasi-Newton curvature, and stochastic gradient methods for noisy, minibatch-scale objectives.

## Notebooks

| Notebook | Covers |
|---|---|
| [`admm_lasso_and_consensus.ipynb`](admm_lasso_and_consensus.ipynb) | ADMM derived from the augmented Lagrangian, applied to Lasso (soft-thresholding derived from the subdifferential) and to distributed global-consensus least squares |
| [`lbfgs_quasi_newton.ipynb`](lbfgs_quasi_newton.ipynb) | The BFGS secant condition and rank-2 update, the two-loop recursion that applies it in `O(mn)` memory, benchmarked on a 100-dimensional Rosenbrock function against gradient descent and full BFGS |
| [`sgd_momentum_adam.ipynb`](sgd_momentum_adam.ipynb) | SGD, heavy-ball momentum (derived from a damped-oscillator ODE discretization), and Adam (derived from bias-corrected moment estimates), isolating momentum's and Adam's distinct failure modes |

## How to build and run

Each notebook is self-contained: open it in Jupyter Notebook/Lab or VS Code and run all cells in
order. Requires `numpy` and `matplotlib`. Every method is implemented directly from its update
equations — no `scipy.optimize`, `cvxpy`, or autodiff library is used for any core method.

## ADMM

Derives the scaled augmented-Lagrangian form and the resulting `x`/`z`/`u` iteration, then applies
it to two structurally different splits. Lasso recovers an 8-feature sparse support exactly from a
60-feature synthetic regression, with the soft-thresholding `z`-update derived directly from the
subdifferential optimality condition rather than quoted; global consensus ADMM splits a
least-squares problem across 5 data shards and its converged global variable matches the
closed-form centralized OLS solution to below `1e-10`. Both problems converge only once the penalty
parameter `rho` is chosen with some care — `rho=1` stalls one to two orders of magnitude short of
the requested tolerance for both problems tested here, while `rho=10` reaches it in a few hundred
iterations, a real (and reported) sensitivity rather than a rounding-level detail.

## L-BFGS

Derives the BFGS secant condition and the resulting rank-2 inverse-Hessian update, then the
two-loop recursion (Nocedal 1980) that applies it using only the last `m` stored `(s,y)` pairs
without ever forming an `n x n` matrix. On a 100-dimensional extended Rosenbrock function, gradient
descent fails to reach a `1e-6` gradient-norm tolerance within a 20,000-iteration budget (stalling
over three orders of magnitude short), while both L-BFGS (`m=10`) and full BFGS reach it in roughly
500 iterations and land within `1e-7` of the true minimizer — L-BFGS matching full BFGS's
convergence almost exactly while storing `O(mn)` floats instead of the `O(n^2)` a dense inverse
Hessian would need.

## SGD, momentum, and Adam

Derives heavy-ball momentum from discretizing a damped-oscillator ODE and Adam's bias correction
from the exact understatement factor `1 - beta^k` in an exponential moving average started at
zero. Two experiments isolate each method's actual effect: on an ill-conditioned quadratic
(condition number 80), momentum reaches a fixed loss target roughly 9x faster than plain SGD early
on, but — reported honestly rather than only in momentum's favor — ends the full run at a *higher*
final loss than plain SGD, because momentum accumulates gradient noise along with signal once the
iterate is already close to the optimum. On a quadratic with a 10,000x scale mismatch between two
coordinates' curvatures, the result is unambiguous: plain SGD's single global step size is forced
to the stiff coordinate's stability limit, leaving the soft coordinate essentially frozen, while
Adam's per-coordinate second-moment normalization reaches a final loss over 800x lower at a step
size two orders of magnitude larger.

## Computational complexity

These three methods exist precisely because the Newton-type solves in `constrained-optimization/`
and `portfolio-construction-black-litterman-sdp/` — $O(n^3)$ per step in the problem dimension
$n$ — stop being affordable as $n$ grows, and each replaces that cost with something that scales
linearly, or near-linearly, in $n$ instead.

**ADMM.** Per iteration: one linear solve against a Cholesky factor that is computed exactly once,
before the iteration begins, and reused throughout — $O(p^2)$ per iteration from the two
triangular back-substitutions (Lasso, $p=60$ features) rather than $O(p^3)$ from refactoring a
system every step, plus an $O(p)$ soft-threshold. Distributed consensus pays this cost once per
shard per iteration ($N=5$ shards, $p_{\dim}=12$), fully parallel across shards, plus an $O(Np_{\dim})$
averaging step to form the shared variable $z$. ADMM's convergence itself is linear (a fixed
fractional reduction in the residual per iteration), which is why the notebook reports the penalty
parameter $\rho$'s effect on iteration count directly rather than treating convergence as
automatic.

**L-BFGS.** The two-loop recursion applies $m$ stored curvature pairs in $O(mn)$ time and memory
— $m=10$, $n=100$ here, so $1{,}000$ floats and flops per iteration against the $O(n^2)=10{,}000$
a dense inverse-Hessian update (full BFGS) would need. Both reach the same $10^{-6}$
gradient-norm tolerance in a comparable number of iterations on the tested Rosenbrock problem
(L-BFGS is not achieving a worse *rate* by discarding curvature history, only using less memory
per step), which is exactly the regime in which L-BFGS is preferred at scale: same near-superlinear
behavior, an order of magnitude less cost per iteration once $n\gg m$.

**SGD, momentum, Adam.** Each is $O(d)$ per iteration in the parameter dimension $d$ — no matrix
formed or factored at any point, only vector arithmetic — which is the entire reason these methods
are the ones used when $d$ is too large for any of the $O(d^2)$-or-worse methods above to be
considered at all. The cost of Adam's per-coordinate second-moment normalization over plain SGD is
a constant multiple of extra vector operations (maintaining $m$ and $v$), not a change in
asymptotic order.
