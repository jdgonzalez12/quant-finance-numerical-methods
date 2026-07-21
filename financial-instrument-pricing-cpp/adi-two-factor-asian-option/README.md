# Two-Factor ADI Scheme for a Fixed-Strike Arithmetic Asian Option

## Problem

A continuously-monitored, fixed-strike arithmetic-average call pays `(A(T) - K)^+` at maturity,
where `A(t) = I(t)/t` is the running average of the underlying and `I(t) = ∫_0^t S(u) du`. Since the
payoff is a functional of the whole path of `S` rather than of `S_T` alone, the option is not
Markovian in `S` by itself; the pair `(S(t), A(t))` is, however, jointly Markov, since
`dI = S dt` (a deterministic, zero-quadratic-variation increment given the path of `S`) implies

```
dA = d(I/t) = (S - A)/t dt        (no diffusion term: A has no independent source of randomness)
```

Applying the Feynman-Kac representation to the pair `(S, A)` under the risk-neutral measure `Q`
(`dS = rS dt + σS dW^Q`) gives the pricing PDE for `V(S, A, t)`:

```
∂V/∂t + (S-A)/t · ∂V/∂A + ½σ²S² ∂²V/∂S² + rS ∂V/∂S − rV = 0,     V(S, A, T) = (A-K)^+
```

This is a genuinely two-dimensional parabolic problem, but of a distinctive kind: the operator is
second-order (diffusive) in `S` and only **first-order (purely advective)** in `A` — `A` inherits no
Brownian source of its own, consistent with `I` being of bounded variation. A direct (non-directional)
implicit discretization would require solving a dense 2D system at every time step; instead, this
project uses a directional **ADI (fractional-step) scheme**, exploiting the fact that each of the two
one-dimensional sub-operators is individually cheap and stable to invert.

## Method

**Splitting.** Writing `L = L_S + L_A` with `L_S V = ½σ²S²V_SS + rSV_S − rV` and
`L_A V = ν·V_A`, `ν ≡ (S-A)/t`, each backward time step (Lie/Yanenko fractional-step splitting) is

```
(I − Δτ L_S) Y  = V^n                     (implicit in S, tridiagonal per fixed A-row)
(I − Δτ L_A) V^{n+1} = Y                  (implicit in A, bidiagonal per fixed S-column)
```

Each sub-step is unconditionally stable on its own (both discrete operators are diagonally dominant
M-matrices, by construction below), so the scheme is stable in both directions; the price paid for
the simplicity of Lie splitting, relative to a symmetric (Strang/Douglas) splitting, is that the
splitting error is `O(Δτ)` rather than `O(Δτ²)` — consistent with, and no worse than, the first-order
time discretization already used in each sub-step.

**Upwinding in `A`, and a sign subtlety worth stating explicitly.** `L_A` is first-order and
hyperbolic, so centered differencing is (weakly) unstable (Duffy, *Financial Instrument Pricing Using
C++*, §21.2.1); an upwind scheme is required. Marching in `τ = T - t` turns
`∂V/∂t + ν V_A = 0` into `∂V/∂τ - ν V_A = 0`, i.e. an advection equation with characteristic speed
`c = -ν` in `τ`-time — the sign is flipped relative to the original `t`-equation. Correct upwinding
looks toward where the characteristic came from: `c > 0` (`ν < 0`) requires a **backward** difference
in `A`, and `c < 0` (`ν ≥ 0`) requires a **forward** difference — the reverse of what naively upwinding
on the sign of `ν` alone would suggest. Both choices give a bidiagonal system with non-positive
off-diagonal entries and unit-or-greater diagonal dominance, i.e. an M-matrix, solved directly with
the Thomas algorithm (no iteration needed, unlike the LCP in the sibling
[`american-option-lcp-psor/`](../american-option-lcp-psor/) project).

**Boundary conditions.** `S = 0`: the operator degenerates to the identity (no diffusion or
convection term survives), handled without special-casing. `S = S_max`: a zero-gradient
(`V_SS ≈ 0`) closure — not the vanilla-call asymptotic `V ~ S - Ke^{-r(T-t)}`, which would be wrong
here, since this payoff depends on the *average* `A`, not on `S` directly, so no value of `V` at
large `S` is implied independent of `A`. `A = 0` and `A = A_max`: zero-gradient as well, the standard
model-agnostic truncation closure.

**A genuine singularity at `t = 0`.** The coefficient `ν = (S-A)/t` blows up as `t → 0`. This is not
a discretization artifact: `A(t)` is a degenerate random variable at `t = 0` (it equals `S_0` with
certainty; the averaging has not yet had time to accumulate any randomness), so the PDE coefficients
are only meaningful for `t > 0`. The price at `t = 0` is recovered as the `t → 0^+` limit, approximated
by stopping the time march one step short, at `t = Δτ`; this introduces an `O(Δτ)` error of the same
order as the scheme's own time-discretization error, so nothing is lost asymptotically. Because `ν`
is large precisely where this matters most (small `t`), the `A`-direction grid must be fine enough to
resolve it well — in practice this problem is considerably more sensitive to the `A`-resolution than
to the `S`-resolution, exactly what the discretization above would predict given that all of the
stiffness (`ν → ∞`) lives in the `A`-operator.

## Computational complexity

Each ADI stage is a batch of independent Thomas solves: Stage 1 runs `N_A+1` solves of size `N_S+1`
(one per fixed `A`-row), Stage 2 runs `N_S+1` solves of size `N_A+1` (one per fixed `S`-column), so one
full time step costs `O(N_S N_A)` — linear in the total number of grid points, not the `O((N_S N_A)^3)`
(or, exploiting 2D bandedness, still substantially worse than linear) a non-directional 2D implicit
solve of the same grid would cost. Total cost over `M` steps is `O(M N_S N_A)`.

`V` and the Stage-1 intermediate `Y` are stored as a single contiguous buffer each (`Grid2D`, row-major
`i * (N_A+1) + k` indexing) rather than as `N_S+1` separately-heap-allocated row vectors
(`vector<vector<double>>`): the latter scatters each row across a different memory block, which costs a
pointer indirection *and* a potential cache miss on every access, in whichever sweep direction is not
the contiguous one (here, that's every access pattern that doesn't walk a single row start-to-end).
Flattening removes `N_S` separate heap allocations per grid and keeps both sweep directions within one
contiguous block. The four Thomas-solve scratch buffers (`sub, diag, sup, rhs`, and the output) are
sized once, to the larger of `N_S+1` and `N_A+1`, and reused for every one of the `(N_S+1)+(N_A+1)`
solves per step, rather than reallocated per solve — at `M-1 = 599` steps and `362` solves per step,
this removes on the order of `2 \times 10^5` vector allocations that the original per-call allocation
pattern would have made.

## Files

| File | Contents |
|---|---|
| `adi_two_factor_asian_option.cpp` | Grid setup, two-stage ADI solver, Monte Carlo cross-check reference |
| `asian_option_value_surface.csv` | `V(S, A)` at `t=0` over the computed grid (subsampled every 2nd node) |

## How to build and run

```
g++ -O2 -std=c++17 -Wall -o adi_two_factor_asian_option adi_two_factor_asian_option.cpp
./adi_two_factor_asian_option
```

## Example output

For `S0 = A0 = 100`, `K = 100`, `r = 0.05`, `σ = 0.20`, `T = 1`:

| Method | Price |
|---|---|
| ADI two-factor PDE (this program) | **5.788540** |
| Monte Carlo, 400,000 paths, 252 monitoring dates (independent cross-check) | 5.789 ± 0.025 (95% CI) |
| Vanilla European call, Black-Scholes closed form | 10.451 |

The ADI price sits inside the Monte Carlo confidence interval, and — as it must, since averaging a
positive-volatility process strictly reduces the variance of the terminal quantity relative to `S_T`
itself — is well below the vanilla European call struck at the same `K`. Convergence is markedly more
sensitive to the `A`-grid than to the `S`-grid: coarsening `N_A` while holding `N_S` fixed moves the
price by several percent, while the reverse does not, matching the fact that all of the operator's
stiffness (the `1/t` singularity as `t → 0`) sits in the `A`-direction term.
