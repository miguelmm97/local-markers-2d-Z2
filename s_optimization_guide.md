# Local S Optimization Guide

This repository now includes a constraint-preserving optimizer for a local,
time-reversal-odd Hermitian involution `S`.  The implementation lives in:

```text
modules/s_optimization.py
```

The tests and benchmark script are:

```text
tests/test_s_optimization.py
scripts/benchmark_s_optimization.py
```

## What the Code Optimizes

The optimizer takes a Hermitian single-particle projector `rho` and searches
for a block-local operator `S` that minimizes

```text
C(S) = -Tr log(I + [S, rho]^2)
```

while preserving these constraints:

```text
S† = S
S² = I
T S T⁻¹ = -S
```

The physical diagnostic reported by the optimizer is not the objective value.
It is the spectral gap

```text
Delta = min(abs(eigvalsh(stilde)))
stilde = rho @ S + S @ rho - S
```

The optimizer selects the best final candidate primarily by this physical gap,
with the objective value used only as a secondary ranking criterion.

## Local Hilbert-Space Convention

The local Hilbert space has dimension `d = 4`.

The basis ordering is:

```text
spin otimes orbital
```

The Pauli matrices `sigma_i` act on spin and `tau_i` act on orbital space.
The local time-reversal unitary is:

```text
U_T = i sigma_y otimes I_2
```

The full-system time-reversal unitary is the block-diagonal repetition of this
local unitary over all sites.

## Constraint-Preserving Parametrization

Instead of adding penalty terms for the constraints, the code parametrizes only
operators that satisfy them.

It builds ten local Hermitian, time-reversal-odd basis matrices:

```text
sigma_i otimes tau_0
sigma_i otimes tau_x
sigma_i otimes tau_z
sigma_0 otimes tau_y
```

where `i = x, y, z`.

For a real parameter vector `theta`, the code constructs:

```text
K(theta) = sum_a theta_a B_a
U(theta) = exp(i K(theta))
s(theta) = U(theta) s0 U(theta)†
s0 = sigma_z otimes I_2
```

Because `K` is Hermitian and time-reversal odd, `U = exp(iK)` commutes with
time reversal.  Therefore `s` remains Hermitian, squares to identity, and is
time-reversal odd up to floating-point precision.

For the global operator, there are two modes:

```text
translation_invariant=True
    S = diag(s, s, ..., s)

translation_invariant=False
    S = diag(s_1, s_2, ..., s_n)
```

The parametrization has redundant coordinates, but that is acceptable for
L-BFGS-B optimization.

## Main API

Typical usage:

```python
from modules.s_optimization import generate_tr_invariant_projector, optimize_local_s

rho = generate_tr_invariant_projector(n_sites=2, seed=1)

result = optimize_local_s(
    rho,
    translation_invariant=True,
    n_restarts=8,
    random_seed=1,
    maxiter=1000,
)

print(result.spectral_gap)
print(result.constraint_residuals)
S = result.S
```

Important fields on the result:

```python
result.S
result.local_operators
result.parameters
result.objective_value
result.spectral_gap
result.success
result.message
result.constraint_residuals
result.diagnostics
```

Useful helpers:

```python
pauli_matrices()
build_tr_odd_basis_d4()
local_time_reversal_unitary()
build_full_time_reversal_unitary(n_sites)
parameters_to_local_operator(...)
assemble_global_operator(...)
compute_stilde(rho, S)
compute_spectral_gap(rho, S)
compute_constraint_residuals(S)
validate_projector(rho, ...)
generate_tr_invariant_projector(n_sites, seed=...)
```

## Optimization Flow

The default `objective="logdet"` flow is:

1. Validate that `rho` is a finite Hermitian projector with dimension divisible
   by four.
2. Generate initial parameter vectors, including the zero vector and random
   Gaussian starts.
3. Run a cheap quadratic warm-start objective:

   ```text
   ||[S, rho]||_F²
   ```

4. Rank warm-start candidates.
5. Refine the best candidates with the exact log-determinant objective.
6. Select the final candidate by largest physical `stilde` gap.
7. Return `S`, local blocks, parameters, residuals, and diagnostics.

The analytic gradients use `scipy.linalg.expm_frechet` to differentiate the
matrix exponential without finite differencing.

## How to Run the Tests

From the repository root:

```bash
pytest tests/test_s_optimization.py -q
```

To run all pytest-discoverable tests:

```bash
pytest -q
```

The current new test suite covers:

- algebraic properties of the ten local basis matrices;
- local parametrization constraints;
- translation-invariant and site-dependent assembly;
- a known commuting case with unit gap;
- random time-reversal-invariant projectors;
- reproducibility for fixed seeds;
- analytic gradient checks against central finite differences;
- invalid input validation;
- result integrity and diagnostic consistency.

## How to Run the Benchmark

Run a small benchmark:

```bash
python scripts/benchmark_s_optimization.py --sites 1 2 --restarts 2 --maxiter 80 --seed 123
```

Run the default benchmark sizes:

```bash
python scripts/benchmark_s_optimization.py
```

Run the site-dependent ansatz:

```bash
python scripts/benchmark_s_optimization.py --sites 1 2 4 --site-dependent
```

Save JSON output:

```bash
python scripts/benchmark_s_optimization.py --sites 1 2 --output results.json
```

The benchmark reports:

```text
n_sites
matrix_dimension
translation_invariant
parameter_count
number_of_restarts
warm_start_runtime
exact_runtime
total_runtime
initial_objective
final_objective
initial_gap
final_gap
iterations
function_evaluations
gradient_evaluations
maximum_constraint_residual
success
```

Timing values are for inspection only.  They are machine-dependent and should
not be treated as strict pass/fail thresholds.

## Dependencies

The new code uses:

```text
numpy
scipy
pytest
```

No environment `.yml` or `.yaml` file was present in this repository, so no
environment file was updated.

## Numerical Notes

The optimization problem is nonconvex.  Multiple restarts improve robustness
but do not guarantee the global optimum.

The log-determinant objective is a smooth surrogate that penalizes small
eigenvalues of `stilde²`.  The physical gap is always computed separately from
`stilde`.

The code clips only tiny negative eigenvalues of `I + [S, rho]^2` caused by
roundoff.  Significant negative eigenvalues are treated as numerical failures
for that candidate.

