# Black-Scholes Option Pricing via the Keller Box Scheme

## Problem

Price a European call `V(S,tau)` under Black-Scholes,

```
dV/dtau = (1/2) sigma^2 S^2 d2V/dS2 + r S dV/dS - r V,   tau = T - t,
V(S,0) = max(S-K, 0),   V(0,tau) = 0,   V(S_max,tau) = S_max - K e^{-r tau}.
```

Crank-Nicolson discretizes this second-order operator directly, evaluating `d2V/dS2` and `dV/dS`
at the same mesh points it evaluates `dV/dtau`. Keller's Box scheme (Keller, 1971) instead reduces
the PDE to an equivalent first-order system and discretizes *that* system on a space-time box,
averaging every quantity to the box's center. The payoff is twofold: both `V` and its gradient come
out second-order accurate simultaneously (Crank-Nicolson is only first-order accurate in the
gradient at the boundary, a point Keller's own analysis emphasizes), and the scheme remains
unconditionally stable for exactly the reason Crank-Nicolson is — it is again an implicit trapezoidal
rule, just applied to a first-order system rather than to the original second-order equation.

## Method

**Reduction to a first-order system.** Introduce the gradient `W := dV/dS` as an independent unknown.
Substituting `d2V/dS2 = dW/dS` into the PDE gives the equivalent pair

```
dV/dS = W                                                                  (I)
(1/2) sigma^2 S^2 dW/dS = dV/dtau - r S W + r V                             (II)
```

with no derivative of order higher than one. This is the direct analogue, for the Black-Scholes
operator specifically, of the reduction Keller carries out for the general self-adjoint parabolic
problem `du/dt = d/dx(a du/dx) + cu + S`; because the convection term `rS dV/dS` here does not
collapse into pure divergence form `(a V_S)_S`, the box-averaging is applied directly to (I)-(II)
rather than to a further-transformed self-adjoint variable, which is the standard way the method is
extended to convection-diffusion-reaction operators in the literature that followed Keller (1971).

**Box discretization.** On the cell spanning `[S_j, S_{j+1}] x [tau_n, tau_{n+1}]`, every unknown is
averaged to the box center. Writing `V_{j+1/2} := (V_j + V_{j+1})/2` (space average) and using a
backward difference in `tau` for the box-centered value, equation (I) becomes

```
(V_{j+1}^{n+1} - V_j^{n+1}) / h = (W_j^{n+1} + W_{j+1}^{n+1}) / 2
```

and equation (II) becomes

```
(1/2) sigma^2 S_{j+1/2}^2 (W_{j+1}^{n+1} - W_j^{n+1}) / h
    = (V_{j+1/2}^{n+1} - V_{j+1/2}^{n}) / dtau - r S_{j+1/2} W_{j+1/2}^{n+1} + r V_{j+1/2}^{n+1}.
```

Two Dirichlet boundary conditions, `V_0 = 0` and `V_J = S_max - K e^{-r tau}`, close the system: with
`J+1` spatial nodes there are `2(J+1)` unknowns `(V_j, W_j)`, and the `J` cells contribute `2J`
equations, exactly matched by the 2 boundary rows. `W_0` and `W_J` are *not* prescribed directly —
they are determined implicitly by the interior box equations that reference them, which is the
correct closure for a pure-Dirichlet problem in `V` alone.

**Solving the system.** The natural sparsity pattern here is not block-*tridiagonal* but
block-*bidiagonal*: because the Box scheme is a first-order system, cell `j`'s two equations couple
only `(V_j, W_j)` to `(V_{j+1}, W_{j+1})` — there is no direct coupling back to `(V_{j-1}, W_{j-1})`,
unlike the block-tridiagonal pattern a central-difference discretization of the original second-order
PDE would give. `solveBoxBlockBidiagonal` exploits this directly: it carries a single reduced scalar
equation `R_j: e_0 V_j + e_1 W_j = e_r` forward (seeded from the `V_0 = 0` boundary condition),
algebraically combines it with cell `j`'s pair of equations to eliminate `(V_j, W_j)` and produce
`R_{j+1}`, and recovers every `(V_j, W_j)` by back-substitution once the last reduced equation meets
the `V_J` boundary condition — the 2x2-block generalization of the scalar Thomas algorithm in
`tridiagonal-solvers/`, and Keller's own block-elimination algorithm for this scheme. This runs in
`O(J)`, replacing the `O(J^3)` a dense solve of the same `2(J+1)`-dimensional system would cost (see
"Computational complexity" below). The dense solver is kept in the code, under the name
`solveDenseReference`, purely as an independent cross-check run once at the end of the program — not
inside the per-timestep hot path — and the two solutions of the identical linear system agree to
`~10^-14`, i.e. floating-point round-off.

## Computational complexity

For `J` spatial cells and `N` time steps, the linear system solved at each step has dimension
`n = 2(J+1)`. A dense solve (partial-pivoted Gaussian elimination) costs `O(n^3) = O(J^3)` per step,
`O(N J^3)` total — for `J=200, N=100` used here, on the order of `10^9` floating-point operations.
`solveBoxBlockBidiagonal` costs `O(J)` per step (a fixed, small number of arithmetic operations per
cell, no matrix ever formed), `O(NJ)` total — on the order of `10^4` operations, a difference of
roughly five orders of magnitude at this grid size, growing without bound as `J` increases. Memory
drops correspondingly from `O(J^2)` (the dense matrix) to `O(J)` (the per-cell coefficient arrays and
the forward-sweep's stored reduction coefficients).

## Files

| File | Contents |
|---|---|
| `keller_box_black_scholes.cpp` | Box-scheme solver, dense linear solve, comparison against the closed-form price |

## How to build and run

```
g++ -O2 -std=c++17 -Wall -o keller_box_black_scholes keller_box_black_scholes.cpp
./keller_box_black_scholes
```

No external dependencies beyond the C++ standard library.

## Example output

For `K=25`, `S_max=50`, `sigma=0.20`, `r=0.0436`, `T=1.0` (the same contract priced in
`crank-nicolson-black-scholes/`, `J=200` cells, `N=100` steps): the Box-scheme price matches the
closed-form Black-Scholes value to within `2.7e-3` over the interior of the grid, and the recovered
gradient `W` rises smoothly from `0` toward `1` as `S` moves through the money — the expected shape
of a call's delta, obtained here as a directly-solved-for quantity rather than a finite difference of
the price computed afterward. The `O(J)` block-bidiagonal solve and the `O(J^3)` dense reference solve,
run end-to-end as two entirely independent simulations of the same system, agree to `max|V_fast -
V_dense| = 1.4e-14` and `max|W_fast - W_dense| = 2.8e-14` — floating-point round-off, not an
approximation.
