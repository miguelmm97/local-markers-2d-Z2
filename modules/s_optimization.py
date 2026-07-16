"""Local optimisation of the auxiliary operator S.

This module follows the same plain procedural style as the other files in
``modules/``.  The public functions at the top are the ones you are expected
to call.  The helper functions at the bottom keep repeated pieces of linear
algebra out of the main routine.

The problem is:

    stilde = rho @ S + S @ rho - S

where ``rho`` is a Hermitian projector.  We look for a local block-diagonal
operator ``S`` such that

    S.conj().T = S
    S @ S = I
    T S T^-1 = -S

and we report the physical gap

    min(abs(eigvalsh(stilde))).

Each site has four internal states ordered as ``spin otimes orbital``.  The
local time-reversal unitary is ``i sigma_y otimes I_2``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Any, Iterable, Literal

import numpy as np
from scipy import linalg
from scipy import optimize


ObjectiveName = Literal["logdet", "quadratic"]


@dataclass(frozen=True)
class ConstraintResiduals:
    """Numerical residuals for the constraints on ``S``."""

    hermiticity: float
    involution: float
    time_reversal_oddness: float
    locality: float

    def as_dict(self) -> dict[str, float]:
        """Return the residuals as a dictionary."""

        return asdict(self)


@dataclass(frozen=True)
class OptimizationResult:
    """Result returned by ``optimize_local_s``."""

    S: np.ndarray
    local_operators: list[np.ndarray]
    parameters: np.ndarray
    objective_value: float
    spectral_gap: float
    success: bool
    message: str
    n_iterations: int
    n_function_evaluations: int
    n_gradient_evaluations: int
    constraint_residuals: ConstraintResiduals
    diagnostics: dict[str, Any]


# ---------------------------------------------------------------------------
# Pauli matrices, time reversal, and local generators
# ---------------------------------------------------------------------------


def pauli_matrices() -> dict[str, np.ndarray]:
    """Return the Pauli matrices as small NumPy arrays."""

    sigma_0 = np.eye(2, dtype=np.complex128)
    sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

    return {
        "0": sigma_0,
        "x": sigma_x,
        "y": sigma_y,
        "z": sigma_z,
    }


def local_time_reversal_unitary() -> np.ndarray:
    """Return ``U_T = i sigma_y otimes I_2`` for one site."""

    pauli = pauli_matrices()
    return np.kron(1j * pauli["y"], pauli["0"])


def build_full_time_reversal_unitary(n_sites: int) -> np.ndarray:
    """Return the full block-diagonal time-reversal unitary."""

    if not isinstance(n_sites, int) or n_sites <= 0:
        raise ValueError("n_sites must be a positive integer.")

    return np.kron(
        np.eye(n_sites, dtype=np.complex128),
        local_time_reversal_unitary(),
    )


def build_tr_odd_basis_d4() -> np.ndarray:
    """Return the ten local Hermitian, time-reversal-odd generators.

    The local basis is ordered as ``spin otimes orbital``.  The generators are

    ``sigma_i otimes tau_0``, ``sigma_i otimes tau_x``,
    ``sigma_i otimes tau_z`` for ``i=x,y,z``, and ``sigma_0 otimes tau_y``.
    """

    pauli = pauli_matrices()

    basis = []
    for spin_label in ("x", "y", "z"):
        for orbital_label in ("0", "x", "z"):
            basis.append(np.kron(pauli[spin_label], pauli[orbital_label]))

    basis.append(np.kron(pauli["0"], pauli["y"]))

    return np.asarray(basis, dtype=np.complex128)


def reference_local_operator() -> np.ndarray:
    """Return ``s0 = sigma_z otimes I_2``."""

    pauli = pauli_matrices()
    return np.kron(pauli["z"], pauli["0"]).astype(np.complex128)


# ---------------------------------------------------------------------------
# Building S
# ---------------------------------------------------------------------------


def parameters_to_local_operator(
    parameters: np.ndarray,
    basis: np.ndarray | None = None,
    reference_operator: np.ndarray | None = None,
) -> np.ndarray:
    """Convert ten real parameters into one constrained local operator.

    The construction is

        K = sum_a theta_a B_a
        U = exp(i K)
        s = U s0 U.conj().T

    Because the ``B_a`` are Hermitian and time-reversal odd, this preserves
    Hermiticity, ``s @ s = I``, and time-reversal oddness.
    """

    basis = build_tr_odd_basis_d4() if basis is None else np.asarray(basis)
    reference_operator = (
        reference_local_operator()
        if reference_operator is None
        else np.asarray(reference_operator, dtype=np.complex128)
    )
    parameters = np.asarray(parameters, dtype=float)

    if parameters.shape != (10,):
        raise ValueError("parameters must have shape (10,) for one local block.")
    if basis.shape != (10, 4, 4):
        raise ValueError("basis must have shape (10, 4, 4).")
    if reference_operator.shape != (4, 4):
        raise ValueError("reference_operator must have shape (4, 4).")

    local_operator, _ = _local_operator_and_derivatives(
        parameters,
        basis,
        reference_operator,
        need_derivatives=False,
    )

    return local_operator


def assemble_global_operator(
    local_operators: Iterable[np.ndarray] | np.ndarray,
    n_sites: int | None = None,
) -> np.ndarray:
    """Build a block-diagonal global ``S`` from local ``4 x 4`` blocks."""

    # Input style 1: one block repeated on every site.
    if isinstance(local_operators, np.ndarray) and local_operators.shape == (4, 4):
        if n_sites is None:
            raise ValueError("n_sites is required when a single local block is supplied.")

        blocks = [
            np.asarray(local_operators, dtype=np.complex128)
            for _ in range(n_sites)
        ]

    # Input style 2: an explicit list of one block per site.
    else:
        blocks = [
            np.asarray(block, dtype=np.complex128)
            for block in local_operators
        ]

        if n_sites is not None and len(blocks) != n_sites:
            raise ValueError("number of local blocks does not agree with n_sites.")

    if len(blocks) == 0:
        raise ValueError("at least one local operator is required.")

    for block in blocks:
        if block.shape != (4, 4):
            raise ValueError("each local operator must have shape (4, 4).")

    dimension = 4 * len(blocks)
    S = np.zeros((dimension, dimension), dtype=np.complex128)

    for site, block in enumerate(blocks):
        first = 4 * site
        S[first : first + 4, first : first + 4] = block

    return S


def compute_stilde(rho: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Return ``stilde = rho @ S + S @ rho - S``."""

    rho = np.asarray(rho, dtype=np.complex128)
    S = np.asarray(S, dtype=np.complex128)

    if rho.ndim != 2 or S.ndim != 2 or rho.shape != S.shape:
        raise ValueError("rho and S must be two-dimensional arrays with the same shape.")

    return _hermitize(rho @ S + S @ rho - S)


def compute_spectral_gap(rho: np.ndarray, S: np.ndarray) -> float:
    """Return the physical gap ``min(abs(eigvalsh(stilde)))``."""

    stilde = compute_stilde(rho, S)
    return float(np.min(np.abs(np.linalg.eigvalsh(stilde))))


def compute_constraint_residuals(
    S: np.ndarray,
    *,
    n_sites: int | None = None,
) -> ConstraintResiduals:
    """Measure Hermiticity, involution, time-reversal oddness, and locality."""

    S = np.asarray(S, dtype=np.complex128)

    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError("S must be a square two-dimensional array.")
    if S.shape[0] % 4 != 0:
        raise ValueError("S dimension must be divisible by four.")

    inferred_sites = S.shape[0] // 4
    if n_sites is None:
        n_sites = inferred_sites
    if n_sites != inferred_sites:
        raise ValueError("n_sites does not agree with S.shape[0] // 4.")

    dimension = S.shape[0]
    identity = np.eye(dimension, dtype=np.complex128)
    time_reversal = build_full_time_reversal_unitary(n_sites)
    scale = max(_frobenius_norm(S), 1.0)

    off_block_part = S.copy()
    for site in range(n_sites):
        first = 4 * site
        off_block_part[first : first + 4, first : first + 4] = 0.0

    hermiticity = _frobenius_norm(S - S.conj().T) / scale
    involution = _frobenius_norm(S @ S - identity) / max(np.sqrt(dimension), 1.0)
    time_reversal_oddness = (
        _frobenius_norm(time_reversal @ S.conj() @ time_reversal.conj().T + S)
        / scale
    )
    locality = _frobenius_norm(off_block_part) / scale

    return ConstraintResiduals(
        hermiticity=float(hermiticity),
        involution=float(involution),
        time_reversal_oddness=float(time_reversal_oddness),
        locality=float(locality),
    )


# ---------------------------------------------------------------------------
# Projector validation and test examples
# ---------------------------------------------------------------------------


def validate_projector(
    rho: np.ndarray,
    *,
    n_sites: int | None = None,
    tolerance: float = 1e-9,
    verify_time_reversal: bool = True,
) -> dict[str, float | int | bool]:
    """Check that ``rho`` is a finite Hermitian projector."""

    rho = np.asarray(rho, dtype=np.complex128)

    # Shape and finiteness checks.
    if rho.ndim != 2:
        raise ValueError("rho must be a two-dimensional square array.")
    if rho.shape[0] != rho.shape[1]:
        raise ValueError("rho must be square.")
    if rho.shape[0] % 4 != 0:
        raise ValueError("rho dimension must be divisible by four.")
    if not np.all(np.isfinite(rho)):
        raise ValueError("rho contains NaN or infinite entries.")

    inferred_sites = rho.shape[0] // 4
    if n_sites is not None and n_sites != inferred_sites:
        raise ValueError("n_sites does not agree with rho.shape[0] // 4.")

    # Projector checks.
    rho_hermiticity = _frobenius_norm(rho - rho.conj().T)
    rho_idempotency = _frobenius_norm(rho @ rho - rho)

    if rho_hermiticity > tolerance:
        raise ValueError(
            f"rho is not Hermitian within tolerance: residual={rho_hermiticity:.3e}."
        )
    if rho_idempotency > tolerance:
        raise ValueError(
            f"rho is not a projector within tolerance: residual={rho_idempotency:.3e}."
        )

    # Optional time-reversal check.
    rho_time_reversal_invariance = np.nan
    if verify_time_reversal:
        time_reversal = build_full_time_reversal_unitary(inferred_sites)
        rho_reversed = time_reversal @ rho.conj() @ time_reversal.conj().T
        rho_time_reversal_invariance = _frobenius_norm(rho_reversed - rho)

        if rho_time_reversal_invariance > tolerance:
            raise ValueError(
                "rho is not time-reversal invariant within tolerance: "
                f"residual={rho_time_reversal_invariance:.3e}."
            )

    eigenvalues = np.linalg.eigvalsh(_hermitize(rho))
    rank = int(np.count_nonzero(eigenvalues > 0.5))

    return {
        "dimension": int(rho.shape[0]),
        "n_sites": int(inferred_sites),
        "rho_hermiticity": float(rho_hermiticity),
        "rho_idempotency": float(rho_idempotency),
        "rho_time_reversal_invariance": float(rho_time_reversal_invariance),
        "rank": rank,
        "time_reversal_checked": bool(verify_time_reversal),
    }


def generate_tr_invariant_projector(
    n_sites: int,
    *,
    seed: int | None = None,
    tolerance: float = 1e-9,
) -> np.ndarray:
    """Generate a random half-filled time-reversal-invariant projector."""

    if not isinstance(n_sites, int) or n_sites <= 0:
        raise ValueError("n_sites must be a positive integer.")

    rng = np.random.default_rng(seed)
    dimension = 4 * n_sites

    random_matrix = rng.normal(size=(dimension, dimension))
    random_matrix = random_matrix + 1j * rng.normal(size=(dimension, dimension))
    random_hermitian = _hermitize(random_matrix)

    time_reversal = build_full_time_reversal_unitary(n_sites)
    random_hermitian_reversed = (
        time_reversal @ random_hermitian.conj() @ time_reversal.conj().T
    )
    tr_hermitian = _hermitize(0.5 * (random_hermitian + random_hermitian_reversed))

    # Occupy the lowest half of the spectrum.
    eigenvalues, eigenvectors = np.linalg.eigh(tr_hermitian)
    occupied_indices = np.argsort(eigenvalues)[: dimension // 2]
    occupied_vectors = eigenvectors[:, occupied_indices]
    rho = occupied_vectors @ occupied_vectors.conj().T

    # Symmetrize the projector and project back onto a clean half-filled space.
    rho_reversed = time_reversal @ rho.conj() @ time_reversal.conj().T
    rho = _hermitize(0.5 * (rho + rho_reversed))

    rho_values, rho_vectors = np.linalg.eigh(rho)
    occupied_indices = np.argsort(rho_values)[-dimension // 2 :]
    occupied_vectors = rho_vectors[:, occupied_indices]
    rho = _hermitize(occupied_vectors @ occupied_vectors.conj().T)

    validate_projector(
        rho,
        n_sites=n_sites,
        tolerance=max(tolerance, 1e-8),
        verify_time_reversal=True,
    )

    return rho


# ---------------------------------------------------------------------------
# Cost and gradient API
# ---------------------------------------------------------------------------


def evaluate_objective_and_gradient(
    rho: np.ndarray,
    parameters: np.ndarray,
    *,
    n_sites: int | None = None,
    translation_invariant: bool = True,
    objective: ObjectiveName = "logdet",
    barrier_epsilon: float = 1e-12,
) -> tuple[float, np.ndarray]:
    """Evaluate a cost and its analytic gradient.

    This is mainly useful for testing the gradient.  The main user-facing
    routine is ``optimize_local_s``.
    """

    rho = np.asarray(rho, dtype=np.complex128)
    inferred_sites = _infer_n_sites(rho)

    if n_sites is None:
        n_sites = inferred_sites
    if n_sites != inferred_sites:
        raise ValueError("n_sites does not agree with rho.shape[0] // 4.")

    flat_parameters = _to_flat_parameters(parameters, n_sites, translation_invariant)
    basis = build_tr_odd_basis_d4()
    reference = reference_local_operator()

    value, gradient, _, _, _ = _cost_gradient_s_and_locals(
        flat_parameters,
        rho,
        n_sites,
        translation_invariant,
        objective,
        barrier_epsilon,
        basis,
        reference,
    )

    return value, _from_flat_parameters(gradient, n_sites, translation_invariant)


# ---------------------------------------------------------------------------
# Main optimiser
# ---------------------------------------------------------------------------


def optimize_local_s(
    rho: np.ndarray,
    *,
    n_sites: int | None = None,
    translation_invariant: bool = True,
    objective: ObjectiveName = "logdet",
    warm_start: bool = True,
    n_restarts: int = 8,
    random_seed: int | None = None,
    maxiter: int = 1000,
    tolerance: float = 1e-9,
    barrier_epsilon: float = 1e-12,
    initial_parameters: np.ndarray | None = None,
    validate_input: bool = True,
    verify_time_reversal: bool = True,
    random_scale: float = 0.2,
    collect_history: bool = False,
) -> OptimizationResult:
    """Find a local ``S`` that gives a large gap for ``stilde``.

    The implementation is deliberately direct:

    1. validate inputs;
    2. build starting parameter vectors;
    3. optionally minimise the cheap quadratic cost first;
    4. minimise the requested final cost;
    5. return the finite result with the largest physical gap.
    """

    # ----- option checks -------------------------------------------------
    if objective not in ("logdet", "quadratic"):
        raise ValueError("objective must be either 'logdet' or 'quadratic'.")
    if n_restarts < 1:
        raise ValueError("n_restarts must be at least one.")
    if maxiter < 1:
        raise ValueError("maxiter must be at least one.")
    if barrier_epsilon <= 0:
        raise ValueError("barrier_epsilon must be positive.")
    if random_scale < 0:
        raise ValueError("random_scale must be nonnegative.")

    start_time = time.perf_counter()

    # ----- rho checks ----------------------------------------------------
    rho = np.asarray(rho, dtype=np.complex128)
    inferred_sites = _infer_n_sites(rho)

    if n_sites is None:
        n_sites = inferred_sites
    if n_sites != inferred_sites:
        raise ValueError("n_sites does not agree with rho.shape[0] // 4.")

    rho_diagnostics: dict[str, Any] = {}
    if validate_input:
        rho_diagnostics = validate_projector(
            rho,
            n_sites=n_sites,
            tolerance=tolerance,
            verify_time_reversal=verify_time_reversal,
        )

    # ----- common matrices ----------------------------------------------
    basis = build_tr_odd_basis_d4()
    reference = reference_local_operator()

    # ----- starting points ----------------------------------------------
    start_vectors = _make_start_vectors(
        n_sites,
        translation_invariant,
        n_restarts,
        random_seed,
        random_scale,
        initial_parameters,
    )

    zero_vector = start_vectors[0]
    initial_objective, _, initial_S, _, _ = _cost_gradient_s_and_locals(
        zero_vector,
        rho,
        n_sites,
        translation_invariant,
        "logdet",
        barrier_epsilon,
        basis,
        reference,
    )
    initial_gap = compute_spectral_gap(rho, initial_S)

    restart_summaries: list[dict[str, Any]] = []
    warm_start_runtime = 0.0
    exact_runtime = 0.0

    # ----- choose candidate starts --------------------------------------
    if objective == "logdet" and warm_start:
        # candidate_vectors is a list of possible starting points for the
        # final logdet minimisation.  Each entry stores:
        #   parameters     -> a flat parameter vector;
        #   ranking_value  -> the quadratic cost after the cheap pre-stage;
        #   ranking_gap    -> the physical gap at that same point.
        candidate_vectors = []
        warm_maxiter = max(20, min(maxiter // 4, 200))

        for restart_index, start_vector in enumerate(start_vectors):
            # Run the cheap quadratic minimisation from this starting vector.
            #
            # minimizer_result:
            #     the SciPy OptimizeResult.  Its ``x`` field is the best
            #     parameter vector found in this restart.
            # minimization_seconds:
            #     wall-clock time spent in this one restart.
            # minimization_history:
            #     optional list of cost values, only filled when
            #     collect_history=True.
            minimizer_result, minimization_seconds, minimization_history = _minimize_from_start(
                start_vector,
                rho,
                n_sites,
                translation_invariant,
                "quadratic",
                barrier_epsilon,
                basis,
                reference,
                warm_maxiter,
                collect_history,
            )
            warm_start_runtime += minimization_seconds

            # Rebuild S from the final parameters of this restart.  SciPy only
            # returns parameters; this call gives us the final cost, S, and gap.
            value, _, S, _, _ = _cost_gradient_s_and_locals(
                minimizer_result.x,
                rho,
                n_sites,
                translation_invariant,
                "quadratic",
                barrier_epsilon,
                basis,
                reference,
            )
            gap = compute_spectral_gap(rho, S)
            restart_summaries.append(
                _restart_summary(
                    restart_index,
                    "warm_start",
                    minimizer_result,
                    minimization_seconds,
                    minimization_history,
                    value,
                    gap,
                )
            )

            # Keep this restart as a possible final-stage starting point only
            # if both diagnostics are finite.
            if np.isfinite(value) and np.isfinite(gap):
                candidate_vectors.append(
                    {
                        "restart_index": restart_index,
                        "parameters": minimizer_result.x,
                        "ranking_value": value,
                        "ranking_gap": gap,
                    }
                )
    else:
        # If there is no quadratic pre-stage, the original starting vectors are
        # the candidates.  We still evaluate each one so we can discard
        # nonfinite points and sort the rest.
        candidate_vectors = []

        for restart_index, start_vector in enumerate(start_vectors):
            # value is the cost at the unoptimised starting vector.
            # S is the concrete global operator built from that vector.
            value, _, S, _, _ = _cost_gradient_s_and_locals(
                start_vector,
                rho,
                n_sites,
                translation_invariant,
                objective,
                barrier_epsilon,
                basis,
                reference,
            )
            gap = compute_spectral_gap(rho, S)

            # Keep only finite starts.  These are later passed to the final
            # minimisation stage.
            if np.isfinite(value) and np.isfinite(gap):
                candidate_vectors.append(
                    {
                        "restart_index": restart_index,
                        "parameters": start_vector,
                        "ranking_value": value,
                        "ranking_gap": gap,
                    }
                )

    # If every start failed numerically, return a complete result object with
    # success=False instead of raising an exception from inside the optimizer.
    if len(candidate_vectors) == 0:
        return _failure_result(
            rho,
            n_sites,
            translation_invariant,
            zero_vector,
            "No finite candidate was found before final minimisation.",
            start_time,
            rho_diagnostics,
            restart_summaries,
            barrier_epsilon,
            basis,
            reference,
        )

    # Sort candidate starts before the final minimisation.  Lower cost is
    # better for the smooth minimisation problem; if two costs are similar, a
    # larger physical gap is preferred.
    candidate_vectors.sort(
        key=lambda item: (
            item["ranking_value"],
            -item["ranking_gap"],
            item["restart_index"],
        )
    )

    # Refining every restart with the exact logdet cost can be expensive.  The
    # cheap quadratic stage already ranked them, so only refine the best few.
    if objective == "logdet":
        candidate_vectors = candidate_vectors[: max(1, min(3, len(candidate_vectors)))]

    # ----- final minimisation -------------------------------------------
    final_objective_name = "logdet" if objective == "logdet" else "quadratic"

    # These variables will hold the best final result found so far.  "Best"
    # means largest physical gap first, then lowest objective value if the gaps
    # are tied.
    best_S = None
    best_local_operators = None
    best_parameters = None
    best_minimizer_result = None
    best_objective_value = np.inf
    best_gap = -np.inf
    best_notes: dict[str, Any] = {}

    for candidate in candidate_vectors:
        # Run the final minimisation from this candidate's parameters.
        #
        # candidate["parameters"] is either:
        #   - the final vector from the quadratic pre-stage, or
        #   - one of the original starting vectors if no pre-stage was used.
        minimizer_result, minimization_seconds, minimization_history = _minimize_from_start(
            candidate["parameters"],
            rho,
            n_sites,
            translation_invariant,
            final_objective_name,
            barrier_epsilon,
            basis,
            reference,
            maxiter,
            collect_history,
        )
        exact_runtime += minimization_seconds

        # Convert the final parameter vector into concrete output objects:
        # value      -> final value of the minimised cost;
        # S          -> final global block-diagonal operator;
        # locals_    -> one local 4x4 block per site;
        # notes      -> numerical notes from evaluating the cost.
        value, _, S, locals_, notes = _cost_gradient_s_and_locals(
            minimizer_result.x,
            rho,
            n_sites,
            translation_invariant,
            final_objective_name,
            barrier_epsilon,
            basis,
            reference,
        )
        gap = compute_spectral_gap(rho, S)
        restart_summaries.append(
            _restart_summary(
                candidate["restart_index"],
                final_objective_name,
                minimizer_result,
                minimization_seconds,
                minimization_history,
                value,
                gap,
            )
        )

        # Ignore failed numerical candidates.  The optimizer may occasionally
        # test a bad point during its line search; those should not be returned.
        if np.isfinite(value) and np.isfinite(gap):
            candidate_is_better = (
                gap > best_gap
                or (np.isclose(gap, best_gap) and value < best_objective_value)
            )

            if candidate_is_better:
                best_S = S
                best_local_operators = locals_
                best_parameters = minimizer_result.x
                best_minimizer_result = minimizer_result
                best_objective_value = value
                best_gap = gap
                best_notes = notes

    if best_S is None or best_minimizer_result is None or best_parameters is None:
        return _failure_result(
            rho,
            n_sites,
            translation_invariant,
            zero_vector,
            "No finite candidate was found after final minimisation.",
            start_time,
            rho_diagnostics,
            restart_summaries,
            barrier_epsilon,
            basis,
            reference,
        )

    # ----- final diagnostics --------------------------------------------
    # Recompute residuals from the final returned S.  This does not trust the
    # parametrisation blindly; it gives a direct numerical check.
    residuals = compute_constraint_residuals(best_S, n_sites=n_sites)
    max_residual = max(residuals.as_dict().values())
    allowed_residual = max(1e-7, 100 * tolerance)

    # The optimizer may stop for many reasons.  The final success flag is based
    # on the mathematical constraints of S, which is what the caller needs.
    success = bool(max_residual <= allowed_residual)
    if success:
        message = str(best_minimizer_result.message)
    else:
        message = (
            "A finite candidate was found, but the final constraints exceed "
            f"tolerance: max_residual={max_residual:.3e}."
        )

    # stilde_squared is used only as a diagnostic.  The physical gap itself was
    # already computed as min(abs(eigvalsh(stilde))).
    stilde = compute_stilde(rho, best_S)
    stilde_squared = _hermitize(stilde @ stilde)

    # diagnostics collects scalar information that is useful for inspecting a
    # run, but not central enough to be a top-level OptimizationResult field.
    diagnostics = {
        **rho_diagnostics,
        "minimum_eigenvalue_of_stilde_squared": float(
            np.min(np.linalg.eigvalsh(stilde_squared))
        ),
        "initial_objective": float(initial_objective),
        "final_objective": float(best_objective_value),
        "initial_gap": float(initial_gap),
        "final_gap": float(best_gap),
        "runtime_seconds": float(time.perf_counter() - start_time),
        "warm_start_runtime": float(warm_start_runtime),
        "exact_runtime": float(exact_runtime),
        "restart_summaries": restart_summaries,
        "translation_invariant": bool(translation_invariant),
        "parameter_count": int(best_minimizer_result.x.size),
        "objective": objective,
        "warm_start": bool(warm_start and objective == "logdet"),
        **best_notes,
    }

    # Build the public result.  best_parameters is still the flat optimizer
    # vector, so reshape it into (10,) or (n_sites, 10) for the caller.
    return OptimizationResult(
        S=best_S,
        local_operators=best_local_operators,
        parameters=_from_flat_parameters(best_parameters, n_sites, translation_invariant),
        objective_value=float(best_objective_value),
        spectral_gap=float(best_gap),
        success=success,
        message=message,
        n_iterations=int(best_minimizer_result.nit),
        n_function_evaluations=int(best_minimizer_result.nfev),
        n_gradient_evaluations=int(best_minimizer_result.njev),
        constraint_residuals=residuals,
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# Helper functions below this point
# ---------------------------------------------------------------------------


def _local_operator_and_derivatives(
    parameters: np.ndarray,
    basis: np.ndarray,
    reference: np.ndarray,
    *,
    need_derivatives: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one local operator and optionally its ten derivatives."""

    # K is the Hermitian generator made from the ten basis matrices.
    # parameters[a] is the real coefficient multiplying basis[a].
    K = np.tensordot(parameters, basis, axes=(0, 0))

    # U = exp(iK).  Since K is time-reversal odd, U commutes with time reversal.
    exponential_argument = 1j * K
    U = linalg.expm(exponential_argument)

    # local_operator is s(theta) = U s0 U^dagger.
    local_operator = U @ reference @ U.conj().T
    local_operator = _hermitize(local_operator)

    # Some callers only need s(theta), not derivatives with respect to theta.
    if not need_derivatives:
        return local_operator, np.empty((0, 4, 4), dtype=np.complex128)

    # derivatives[a] is d s(theta) / d parameters[a].
    # expm_frechet gives d exp(A)[E], avoiding finite differences.
    derivatives = np.empty((10, 4, 4), dtype=np.complex128)
    for parameter_index, basis_matrix in enumerate(basis):
        dU = linalg.expm_frechet(
            exponential_argument,
            1j * basis_matrix,
            compute_expm=False,
        )
        derivative = dU @ reference @ U.conj().T + U @ reference @ dU.conj().T
        derivatives[parameter_index] = _hermitize(derivative)

    return local_operator, derivatives


def _build_s_and_derivatives(
    flat_parameters: np.ndarray,
    n_sites: int,
    translation_invariant: bool,
    basis: np.ndarray,
    reference: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Build global S, local blocks, and local derivative arrays."""

    # flat_parameters is what scipy.optimize sees.
    # Convert it into one local 10-vector or one 10-vector per site.
    if translation_invariant:
        local_parameter_blocks = flat_parameters.reshape(1, 10)
    else:
        local_parameter_blocks = flat_parameters.reshape(n_sites, 10)

    # local_blocks contains the local s matrices that will become S.
    # derivative_blocks contains ds/dtheta for the matching local blocks.
    local_blocks = []
    derivative_blocks = []

    # Build every independent local block.  Translation-invariant mode only has
    # one independent block; site-dependent mode has n_sites independent blocks.
    for local_parameters in local_parameter_blocks:
        local_operator, derivatives = _local_operator_and_derivatives(
            local_parameters,
            basis,
            reference,
            need_derivatives=True,
        )
        local_blocks.append(local_operator)
        derivative_blocks.append(derivatives)

    # In translation-invariant mode, repeat the one local operator across all
    # sites.  The derivative is not repeated here; the gradient code sums site
    # contributions into the one shared derivative block.
    if translation_invariant:
        local_blocks_for_all_sites = [local_blocks[0].copy() for _ in range(n_sites)]
    else:
        local_blocks_for_all_sites = local_blocks

    # Assemble the full block-diagonal matrix S.
    S = assemble_global_operator(local_blocks_for_all_sites)

    return S, local_blocks_for_all_sites, derivative_blocks


def _cost_gradient_s_and_locals(
    flat_parameters: np.ndarray,
    rho: np.ndarray,
    n_sites: int,
    translation_invariant: bool,
    objective: ObjectiveName,
    barrier_epsilon: float,
    basis: np.ndarray,
    reference: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, list[np.ndarray], dict[str, Any]]:
    """Return cost, gradient, S, local blocks, and diagnostic notes."""

    # flat_parameters is the real vector being moved by scipy.optimize.
    flat_parameters = np.asarray(flat_parameters, dtype=float)

    # Reject unsupported cost names at this internal boundary too.
    if objective not in ("logdet", "quadratic"):
        raise ValueError("objective must be either 'logdet' or 'quadratic'.")

    # These checks protect scipy's line search from NaNs or wildly large
    # parameters.  A large finite cost is returned instead of raising.
    if not np.all(np.isfinite(flat_parameters)):
        return _bad_cost_result(
            flat_parameters,
            rho,
            n_sites,
            translation_invariant,
            basis,
            reference,
            "nonfinite parameters",
        )

    if np.linalg.norm(flat_parameters) > 1e6:
        return _bad_cost_result(
            flat_parameters,
            rho,
            n_sites,
            translation_invariant,
            basis,
            reference,
            "parameter norm too large",
        )

    # Build the current global S and all local derivatives ds/dtheta.
    S, local_blocks, derivative_blocks = _build_s_and_derivatives(
        flat_parameters,
        n_sites,
        translation_invariant,
        basis,
        reference,
    )

    # The commutator measures how much S mixes occupied and unoccupied states.
    commutator = S @ rho - rho @ S

    # Compute the chosen cost and the gradient with respect to the matrix S.
    # matrix_gradient is d cost / dS, not yet d cost / dtheta.
    if objective == "quadratic":
        value = float(np.real(np.vdot(commutator, commutator)))
        matrix_gradient = -2.0 * (rho @ commutator - commutator @ rho)
        notes: dict[str, Any] = {}
    else:
        value, matrix_gradient, notes = _logdet_value_and_matrix_gradient(
            rho,
            commutator,
            barrier_epsilon,
        )

        if "invalid_objective_reason" in notes:
            return _bad_cost_result(
                flat_parameters,
                rho,
                n_sites,
                translation_invariant,
                basis,
                reference,
                notes["invalid_objective_reason"],
            )

    # If numerical issues occurred, return a large finite cost so the optimizer
    # can reject the point.
    if not np.isfinite(value) or not np.all(np.isfinite(matrix_gradient)):
        return _bad_cost_result(
            flat_parameters,
            rho,
            n_sites,
            translation_invariant,
            basis,
            reference,
            "nonfinite objective or gradient",
        )

    # Chain rule: convert d cost / dS into d cost / dtheta.
    gradient = _parameter_gradient_from_matrix_gradient(
        matrix_gradient,
        derivative_blocks,
        n_sites,
        translation_invariant,
    )

    return value, gradient, S, local_blocks, notes


def _logdet_value_and_matrix_gradient(
    rho: np.ndarray,
    commutator: np.ndarray,
    barrier_epsilon: float,
) -> tuple[float, np.ndarray, dict[str, Any]]:
    """Return the exact logdet cost and matrix gradient dC/dS."""

    # X is the matrix inside the log determinant:
    #     X = I + [S, rho]^2.
    # For valid inputs this equals stilde^2.
    identity = np.eye(rho.shape[0], dtype=np.complex128)
    X = identity + commutator @ commutator
    X = _hermitize(X)

    # Diagonalize X because it should be Hermitian.  This is more stable than
    # computing a determinant directly.
    eigenvalues, eigenvectors = np.linalg.eigh(X)
    minimum_eigenvalue = float(np.min(eigenvalues))

    # notes are diagnostic values copied into result.diagnostics.
    notes: dict[str, Any] = {
        "minimum_eigenvalue_of_X": minimum_eigenvalue,
        "clipped_x_eigenvalues": int(np.count_nonzero(eigenvalues < 0.0)),
    }

    # A significantly negative eigenvalue means the formula has broken down.
    # Small negative values can happen from floating-point roundoff and are
    # clipped below.
    if minimum_eigenvalue < -1e-8:
        return 1.0e100, np.zeros_like(rho), {
            **notes,
            "invalid_objective_reason": "significant negative X eigenvalue",
        }

    # Add a small positive epsilon before the log so the cost remains finite
    # near a closing gap.
    clipped = np.clip(eigenvalues, 0.0, None)
    shifted = clipped + barrier_epsilon

    value = float(-np.sum(np.log(shifted)))

    # R = (X + epsilon I)^-1 from eigendecomposition, not explicit inverse.
    inverse_shifted = 1.0 / shifted
    R = (eigenvectors * inverse_shifted) @ eigenvectors.conj().T

    # Matrix gradient of the logdet cost:
    #     dC/dS = -2 [rho, R [S, rho]].
    RA = R @ commutator
    matrix_gradient = -2.0 * (rho @ RA - RA @ rho)

    return value, matrix_gradient, notes


def _parameter_gradient_from_matrix_gradient(
    matrix_gradient: np.ndarray,
    derivative_blocks: list[np.ndarray],
    n_sites: int,
    translation_invariant: bool,
) -> np.ndarray:
    """Contract the matrix gradient with local derivatives."""

    # translation_invariant=True means all sites share one ten-parameter local
    # operator.  The gradient is therefore one vector of length 10.
    if translation_invariant:
        gradient = np.zeros(10, dtype=float)
        derivatives = derivative_blocks[0]

        for site in range(n_sites):
            # Only the 4x4 diagonal block of dC/dS contributes, because the
            # allowed dS/dtheta is block diagonal and local.
            first = 4 * site
            local_matrix_gradient = matrix_gradient[first : first + 4, first : first + 4]

            for parameter_index in range(10):
                # Hilbert-Schmidt inner product:
                #   dC/dtheta = Re Tr((dC/dS)^dagger (dS/dtheta)).
                gradient[parameter_index] += float(
                    np.real(np.vdot(local_matrix_gradient, derivatives[parameter_index]))
                )

        return gradient

    # Site-dependent mode has one separate ten-parameter gradient per site.
    site_gradients = []
    for site in range(n_sites):
        first = 4 * site
        local_matrix_gradient = matrix_gradient[first : first + 4, first : first + 4]
        derivatives = derivative_blocks[site]

        local_gradient = np.zeros(10, dtype=float)
        for parameter_index in range(10):
            # Same local Hilbert-Schmidt contraction, but stored for this site
            # only instead of summed over all sites.
            local_gradient[parameter_index] = float(
                np.real(np.vdot(local_matrix_gradient, derivatives[parameter_index]))
            )

        site_gradients.append(local_gradient)

    return np.concatenate(site_gradients)


def _bad_cost_result(
    flat_parameters: np.ndarray,
    rho: np.ndarray,
    n_sites: int,
    translation_invariant: bool,
    basis: np.ndarray,
    reference: np.ndarray,
    reason: str,
) -> tuple[float, np.ndarray, np.ndarray, list[np.ndarray], dict[str, Any]]:
    """Return a large finite cost for a bad optimizer probe."""

    # Try to build a concrete S for diagnostics.  If parameters are nonfinite,
    # replace them with zero just for this diagnostic construction.
    safe_parameters = np.asarray(flat_parameters, dtype=float).copy()
    if not np.all(np.isfinite(safe_parameters)):
        safe_parameters = np.zeros_like(safe_parameters)

    # Build S from safe parameters so failure results still contain an operator
    # whose residuals can be inspected.
    S, local_blocks, _ = _build_s_and_derivatives(
        safe_parameters,
        n_sites,
        translation_invariant,
        basis,
        reference,
    )

    # The zero gradient is not meant to be physically meaningful here.  It is
    # paired with a huge cost so the optimizer treats this point as unusable.
    return (
        1.0e100,
        np.zeros_like(safe_parameters),
        S,
        local_blocks,
        {"invalid_objective_reason": reason},
    )


def _minimize_from_start(
    start_vector: np.ndarray,
    rho: np.ndarray,
    n_sites: int,
    translation_invariant: bool,
    objective: ObjectiveName,
    barrier_epsilon: float,
    basis: np.ndarray,
    reference: np.ndarray,
    maxiter: int,
    collect_history: bool,
) -> tuple[optimize.OptimizeResult, float, list[float]]:
    """Run one L-BFGS-B minimisation."""

    # history stores the cost after each optimizer iteration.  It stays empty
    # unless the caller explicitly asks for it.
    history: list[float] = []

    def value_and_gradient(current_vector: np.ndarray) -> tuple[float, np.ndarray]:
        # SciPy calls this many times.  It must return exactly two objects:
        # the scalar cost and the gradient with respect to the flat parameters.
        value, gradient, _, _, _ = _cost_gradient_s_and_locals(
            current_vector,
            rho,
            n_sites,
            translation_invariant,
            objective,
            barrier_epsilon,
            basis,
            reference,
        )
        return value, gradient

    def callback(current_vector: np.ndarray) -> None:
        # The callback is only for optional diagnostics.  The optimization does
        # not depend on this history.
        if not collect_history:
            return

        value, _, _, _, _ = _cost_gradient_s_and_locals(
            current_vector,
            rho,
            n_sites,
            translation_invariant,
            objective,
            barrier_epsilon,
            basis,
            reference,
        )
        history.append(float(value))

    # L-BFGS-B is the only optimizer used here.  There are no parameter bounds:
    # the matrix exponential already maps every real vector to a valid local S.
    start_time = time.perf_counter()
    result = optimize.minimize(
        value_and_gradient,
        np.asarray(start_vector, dtype=float),
        method="L-BFGS-B",
        jac=True,
        callback=callback,
        options={
            "maxiter": int(maxiter),
            "ftol": 1e-12,
            "gtol": 1e-8,
            "maxls": 40,
        },
    )
    seconds = time.perf_counter() - start_time

    return result, seconds, history


def _restart_summary(
    restart_index: int,
    stage: str,
    result: optimize.OptimizeResult,
    seconds: float,
    history: list[float],
    value: float,
    gap: float,
) -> dict[str, Any]:
    """Make a small summary dictionary for one restart."""

    # This dictionary is kept simple so it can be printed or written as JSON.
    # result.x is not stored here because it can be large; diagnostics keep only
    # its norm.
    summary = {
        "restart_index": int(restart_index),
        "stage": stage,
        "success": bool(result.success),
        "message": str(result.message),
        "objective_value": float(value),
        "spectral_gap": float(gap),
        "iterations": int(result.nit),
        "function_evaluations": int(result.nfev),
        "gradient_evaluations": int(result.njev),
        "runtime_seconds": float(seconds),
        "parameter_norm": float(np.linalg.norm(result.x)),
    }

    if history:
        summary["history"] = history

    return summary


def _make_start_vectors(
    n_sites: int,
    translation_invariant: bool,
    n_restarts: int,
    random_seed: int | None,
    random_scale: float,
    initial_parameters: np.ndarray | None,
) -> list[np.ndarray]:
    """Make zero, user-supplied, and random starting vectors."""

    if translation_invariant:
        parameter_count = 10
    else:
        parameter_count = 10 * n_sites

    start_vectors = [np.zeros(parameter_count, dtype=float)]

    if initial_parameters is not None:
        start_vectors.extend(
            _normalize_initial_parameters(initial_parameters, n_sites, translation_invariant)
        )

    rng = np.random.default_rng(random_seed)
    while len(start_vectors) < n_restarts:
        start_vectors.append(rng.normal(scale=random_scale, size=parameter_count))

    return start_vectors


def _normalize_initial_parameters(
    initial_parameters: np.ndarray,
    n_sites: int,
    translation_invariant: bool,
) -> list[np.ndarray]:
    """Validate and flatten user-supplied initial parameters."""

    parameters = np.asarray(initial_parameters, dtype=float)
    if not np.all(np.isfinite(parameters)):
        raise ValueError("initial_parameters must contain only finite values.")

    if translation_invariant:
        if parameters.shape == (10,):
            return [parameters.copy()]
        if parameters.ndim == 2 and parameters.shape[1] == 10:
            return [row.copy() for row in parameters]

        raise ValueError(
            "translation-invariant initial_parameters must have shape (10,) "
            "or (n_initial, 10)."
        )

    if parameters.shape == (n_sites, 10):
        return [parameters.reshape(-1).copy()]
    if parameters.ndim == 3 and parameters.shape[1:] == (n_sites, 10):
        return [block.reshape(-1).copy() for block in parameters]

    raise ValueError(
        "site-dependent initial_parameters must have shape (n_sites, 10) "
        "or (n_initial, n_sites, 10)."
    )


def _to_flat_parameters(
    parameters: np.ndarray,
    n_sites: int,
    translation_invariant: bool,
) -> np.ndarray:
    """Convert public parameter shape to optimizer vector shape."""

    parameters = np.asarray(parameters, dtype=float)

    if translation_invariant:
        if parameters.shape != (10,):
            raise ValueError("translation-invariant parameters must have shape (10,).")
        return parameters.copy()

    if parameters.shape != (n_sites, 10):
        raise ValueError("site-dependent parameters must have shape (n_sites, 10).")

    return parameters.reshape(-1).copy()


def _from_flat_parameters(
    flat_parameters: np.ndarray,
    n_sites: int,
    translation_invariant: bool,
) -> np.ndarray:
    """Convert optimizer vector shape back to public parameter shape."""

    flat_parameters = np.asarray(flat_parameters, dtype=float)

    if translation_invariant:
        return flat_parameters.reshape(10).copy()

    return flat_parameters.reshape(n_sites, 10).copy()


def _failure_result(
    rho: np.ndarray,
    n_sites: int,
    translation_invariant: bool,
    flat_parameters: np.ndarray,
    message: str,
    start_time: float,
    rho_diagnostics: dict[str, Any],
    restart_summaries: list[dict[str, Any]],
    barrier_epsilon: float,
    basis: np.ndarray,
    reference: np.ndarray,
) -> OptimizationResult:
    """Return a complete result object when no valid candidate is found."""

    value, _, S, locals_, _ = _cost_gradient_s_and_locals(
        flat_parameters,
        rho,
        n_sites,
        translation_invariant,
        "logdet",
        barrier_epsilon,
        basis,
        reference,
    )

    residuals = compute_constraint_residuals(S, n_sites=n_sites)
    diagnostics = {
        **rho_diagnostics,
        "runtime_seconds": float(time.perf_counter() - start_time),
        "restart_summaries": restart_summaries,
    }

    return OptimizationResult(
        S=S,
        local_operators=locals_,
        parameters=_from_flat_parameters(flat_parameters, n_sites, translation_invariant),
        objective_value=float(value),
        spectral_gap=compute_spectral_gap(rho, S),
        success=False,
        message=message,
        n_iterations=0,
        n_function_evaluations=0,
        n_gradient_evaluations=0,
        constraint_residuals=residuals,
        diagnostics=diagnostics,
    )


def _infer_n_sites(rho: np.ndarray) -> int:
    """Infer the number of local four-dimensional sites from ``rho``."""

    rho = np.asarray(rho)

    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("rho must be a square two-dimensional array.")
    if rho.shape[0] % 4 != 0:
        raise ValueError("rho dimension must be divisible by four.")

    return rho.shape[0] // 4


def _hermitize(matrix: np.ndarray) -> np.ndarray:
    """Return ``0.5 * (matrix + matrix.conj().T)``."""

    return 0.5 * (matrix + matrix.conj().T)


def _frobenius_norm(matrix: np.ndarray) -> float:
    """Return the Frobenius norm as a plain float."""

    return float(np.linalg.norm(matrix, ord="fro"))
