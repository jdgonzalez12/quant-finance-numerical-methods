# Tridiagonal Linear Systems: LU Factorization and Godunov's Double Sweep

## Problem

Solve the linear system `Au = f` where `A in R^{J-1 x J-1}` is tridiagonal,

```
a_j u_{j-1} + b_j u_j + c_j u_{j+1} = f_j,   j = 1, ..., J-1,   u_0 = phi, u_J = psi
```

This is the linear-algebra kernel underneath every implicit finite-difference scheme in this
repository: a three-point discretization of a second-order spatial operator produces exactly this
band structure, and the same primitive reappears as the per-direction solve inside every ADI
splitting (`crank-nicolson-heston/`) and every fully implicit time step (`crank-nicolson-black-scholes/`,
`keller-box-black-scholes/`).

## Method

**Well-posedness.** `A` need not be symmetric or positive definite for these algorithms to apply;
the relevant sufficient condition is (weak) diagonal dominance,

```
|b_j| >= |a_j| + |c_j|   for every j,   with strict inequality for at least one j,
```

together with `a_j, c_j != 0`. Under this hypothesis `A` is a nonsingular M-matrix-like operator
(more precisely, an irreducibly diagonally dominant matrix), which is the standard sufficient
condition guaranteeing that Gaussian elimination *without pivoting* never divides by a vanishing
pivot and remains backward stable — this is exactly the regime three-point discretizations of
parabolic and elliptic operators with non-negative diffusion coefficients land in, which is why
finance and scientific-computing codebases almost never pivot tridiagonal solves.

**Method 1 — LU factorization.** Writing `A = LU` with `L` unit-lower-bidiagonal and `U`
unit-upper-bidiagonal on the tridiagonal band gives, by direct comparison of entries, the `O(J)`
recursion

```
beta_1 = b_1,                              gamma_1 = c_1 / beta_1
beta_j = b_j - a_j*gamma_{j-1},             gamma_j = c_j / beta_j     (j = 2, ..., J-2)
beta_{J-1} = b_{J-1} - a_{J-1}*gamma_{J-2}
```

followed by the forward solve of `Lz = f` and the backward solve of `Uu = z`. Diagonal dominance of
`A` propagates to a uniform lower bound `|beta_j| >= |a_j|` at every step (an elementary induction),
which is precisely the statement that the factorization cannot break down and that the forward/backward
sweeps are themselves diagonally dominant, hence stable.

**Method 2 — Godunov's double sweep (balayage).** Rather than eliminate row-by-row, this method posits
that the solution has the affine form

```
u_j = L_{j+1/2} u_{j+1} + K_{j+1/2},   L_{1/2} = 0,   K_{1/2} = phi,
```

and derives, from substituting this ansatz into the governing three-term recurrence, the coupled
`O(J)` recursion for the two auxiliary sequences

```
L_{j+1/2} = -c_j / (b_j + a_j L_{j-1/2}),     K_{j+1/2} = (f_j - a_j K_{j-1/2}) / (b_j + a_j L_{j-1/2}).
```

Boundary data enters as the *seed* of the forward sweep rather than through a modified
right-hand side, which is why this parametrization (standard in the Soviet/Russian numerical PDE
literature under the name *balayage*, and widely used for two-point boundary value problems with
Dirichlet data) is often preferred when the same tridiagonal shape recurs across many right-hand
sides with varying boundary values: `a_j, b_j, c_j` are precomputed once and only `phi, psi, f`
change per solve. Algebraically, `L_{j+1/2} = -gamma_j` and `K_{j+1/2} = z_j` in the LU notation
above — the two methods are the same factorization written in two different bases, so their outputs
agree to floating-point round-off whenever both are well-posed; this identity is exactly what the
tests below verify.

## Computational complexity

Both methods do a fixed, small number of arithmetic operations (a handful of multiplications,
divisions, and subtractions) per index `j`, with no operation ever touching two indices more than one
apart — so both are `O(J)` in time and `O(J)` in (auxiliary) memory, for a system of `J-1` unknowns.
This is the entire reason tridiagonal solves are the load-bearing primitive of every implicit scheme
in this repository: a general dense solve of the same system is `O(J^3)` in time and `O(J^2)` in
memory, so replacing it with either method here is not a constant-factor tweak but an asymptotic
change, and it is what keeps every implicit finite-difference scheme elsewhere in this folder
(`keller-box-black-scholes/`, `exponentially-fitted-scheme/`, `american-option-lcp-psor/`,
`adi-two-factor-asian-option/`) linear-cost per time step in the number of spatial nodes. Both
solvers here also take their input coefficient vectors by `const&` and write only into the routine's
own output buffer, so a single solve allocates exactly the arrays it returns — nothing beyond that
is copied or reallocated per call.

## Files

| File | Contents |
|---|---|
| `tridiagonal_solvers.cpp` | Both solvers, three validation problems, and the diagonal-dominance discussion above realized in code |

## How to build and run

```
g++ -O2 -std=c++17 -Wall -o tridiagonal_solvers tridiagonal_solvers.cpp
./tridiagonal_solvers
```

No external dependencies beyond the C++ standard library.

## Example output

**Test 1** (`u'' + u = 0` on `(0,1)`, `u(0)=0`, `u(1)=1`, exact solution `u(x) = sin(x)/sin(1)`,
`J=20`): both methods agree with each other to floating-point round-off and with the exact solution
to `O(h^2) ~ 1.65e-5`, the expected truncation error of the central-difference discretization.

**Test 2** (random diagonally dominant systems, `n` up to 5000): the LU solver's own residual
`max|Au - f|` is at machine-epsilon level for every size tested, confirming `O(J)` scaling introduces
no accumulating error.

**Test 3** (`u'' - u = -1` on `(0,1)`, `u(0)=u(1)=0`, exact solution
`u(x) = 1 - cosh(x-1/2)/cosh(1/2)`, `J=40`): LU and the double sweep again agree exactly with each
other and match the closed form to `O(h^2)`.
