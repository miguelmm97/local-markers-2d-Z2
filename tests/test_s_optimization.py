import numpy as np
import pytest

from modules.s_optimization import (
    assemble_global_operator,
    build_full_time_reversal_unitary,
    build_tr_odd_basis_d4,
    compute_constraint_residuals,
    compute_spectral_gap,
    evaluate_objective_and_gradient,
    generate_tr_invariant_projector,
    local_time_reversal_unitary,
    optimize_local_s,
    parameters_to_local_operator,
    pauli_matrices,
    reference_local_operator,
    validate_projector,
)


def test_tr_odd_basis_is_hermitian_tr_odd_and_independent():
    basis = build_tr_odd_basis_d4()
    time_reversal_unitary = local_time_reversal_unitary()

    assert basis.shape == (10, 4, 4)
    for basis_matrix in basis:
        assert np.allclose(basis_matrix, basis_matrix.conj().T)
        transformed = time_reversal_unitary @ basis_matrix.conj() @ time_reversal_unitary.conj().T
        assert np.allclose(transformed, -basis_matrix)

    realified = np.stack(
        [np.concatenate((matrix.real.ravel(), matrix.imag.ravel())) for matrix in basis]
    )
    assert np.linalg.matrix_rank(realified, tol=1e-12) == 10


def test_local_parametrization_preserves_all_constraints():
    rng = np.random.default_rng(123)
    time_reversal_unitary = local_time_reversal_unitary()

    for _ in range(4):
        parameters = rng.normal(scale=0.3, size=10)
        local_s = parameters_to_local_operator(parameters)

        assert np.allclose(local_s, local_s.conj().T, atol=1e-12)
        assert np.allclose(local_s @ local_s, np.eye(4), atol=1e-12)
        transformed = time_reversal_unitary @ local_s.conj() @ time_reversal_unitary.conj().T
        assert np.allclose(transformed, -local_s, atol=1e-12)


def test_translation_invariant_and_site_dependent_assembly():
    rng = np.random.default_rng(7)
    local_s = parameters_to_local_operator(rng.normal(scale=0.1, size=10))
    global_s = assemble_global_operator(local_s, n_sites=3)

    assert global_s.shape == (12, 12)
    assert compute_constraint_residuals(global_s).locality == pytest.approx(0.0)
    for site in range(3):
        start = 4 * site
        assert np.allclose(global_s[start : start + 4, start : start + 4], local_s)

    blocks = [parameters_to_local_operator(rng.normal(scale=0.1, size=10)) for _ in range(2)]
    site_dependent_s = assemble_global_operator(blocks)

    assert site_dependent_s.shape == (8, 8)
    assert np.allclose(site_dependent_s[:4, :4], blocks[0])
    assert np.allclose(site_dependent_s[4:, 4:], blocks[1])


def test_known_commuting_projector_has_unit_gap():
    pauli = pauli_matrices()
    orbital_projector = np.array([[1, 0], [0, 0]], dtype=np.complex128)
    local_projector = np.kron(pauli["0"], orbital_projector)
    rho = assemble_global_operator(local_projector, n_sites=2)

    result = optimize_local_s(
        rho,
        n_restarts=2,
        random_seed=11,
        maxiter=80,
        tolerance=1e-8,
    )

    commutator_norm = np.linalg.norm(result.S @ rho - rho @ result.S)
    assert result.success, result.message
    assert result.objective_value <= 1e-8
    assert result.spectral_gap >= 1.0 - 1e-8
    assert commutator_norm <= 1e-8


def test_random_projector_optimization_is_reproducible_and_finite():
    rho = generate_tr_invariant_projector(n_sites=1, seed=22)

    result_a = optimize_local_s(
        rho,
        n_restarts=3,
        random_seed=5,
        maxiter=80,
        tolerance=1e-8,
    )
    result_b = optimize_local_s(
        rho,
        n_restarts=3,
        random_seed=5,
        maxiter=80,
        tolerance=1e-8,
    )

    assert result_a.success, result_a.message
    assert np.isfinite(result_a.objective_value)
    assert np.isfinite(result_a.spectral_gap)
    assert result_a.spectral_gap >= 0.0
    assert result_a.diagnostics["final_objective"] <= result_a.diagnostics["initial_objective"] + 1e-8
    assert np.allclose(result_a.parameters, result_b.parameters)
    assert result_a.objective_value == pytest.approx(result_b.objective_value, abs=1e-10)
    assert result_a.spectral_gap == pytest.approx(result_b.spectral_gap, abs=1e-10)


def test_quadratic_gradient_matches_central_finite_difference():
    rho = generate_tr_invariant_projector(n_sites=1, seed=33)
    parameters = np.random.default_rng(44).normal(scale=0.08, size=10)

    _, analytic_gradient = evaluate_objective_and_gradient(
        rho,
        parameters,
        objective="quadratic",
    )
    finite_difference = _central_finite_difference(rho, parameters, objective="quadratic")

    absolute_error = np.max(np.abs(analytic_gradient - finite_difference))
    relative_error = np.linalg.norm(analytic_gradient - finite_difference) / max(
        np.linalg.norm(finite_difference),
        1e-12,
    )
    assert absolute_error < 1e-6
    assert relative_error < 1e-5


def test_exact_gradient_matches_central_finite_difference():
    rho = generate_tr_invariant_projector(n_sites=1, seed=55)
    parameters = np.random.default_rng(66).normal(scale=0.08, size=10)

    _, analytic_gradient = evaluate_objective_and_gradient(
        rho,
        parameters,
        objective="logdet",
        barrier_epsilon=1e-10,
    )
    finite_difference = _central_finite_difference(
        rho,
        parameters,
        objective="logdet",
        barrier_epsilon=1e-10,
    )

    absolute_error = np.max(np.abs(analytic_gradient - finite_difference))
    relative_error = np.linalg.norm(analytic_gradient - finite_difference) / max(
        np.linalg.norm(finite_difference),
        1e-12,
    )
    assert absolute_error < 1e-5
    assert relative_error < 1e-4


def test_validation_failures_are_informative():
    rho = generate_tr_invariant_projector(n_sites=1, seed=77)

    with pytest.raises(ValueError, match="square"):
        validate_projector(np.ones((2, 3)))
    with pytest.raises(ValueError, match="divisible by four"):
        validate_projector(np.eye(6))
    with pytest.raises(ValueError, match="Hermitian"):
        bad = rho.copy()
        bad[0, 1] += 0.1
        validate_projector(bad)
    with pytest.raises(ValueError, match="projector"):
        validate_projector(0.5 * np.eye(4))
    with pytest.raises(ValueError, match="NaN or infinite"):
        bad = rho.copy()
        bad[0, 0] = np.nan
        validate_projector(bad)
    with pytest.raises(ValueError, match="n_sites"):
        validate_projector(rho, n_sites=2)
    with pytest.raises(ValueError, match="initial_parameters"):
        optimize_local_s(rho, initial_parameters=np.zeros((2, 2)), maxiter=5)


def test_result_integrity_uses_returned_operator():
    rho = generate_tr_invariant_projector(n_sites=2, seed=88)

    result = optimize_local_s(
        rho,
        translation_invariant=False,
        n_restarts=2,
        random_seed=99,
        maxiter=60,
        tolerance=1e-8,
    )

    reconstructed = assemble_global_operator(result.local_operators)
    recalculated_residuals = compute_constraint_residuals(result.S)

    assert result.success, result.message
    assert result.parameters.shape == (2, 10)
    assert np.allclose(result.S, reconstructed)
    assert compute_spectral_gap(rho, result.S) == pytest.approx(result.spectral_gap, abs=1e-10)
    assert recalculated_residuals.as_dict() == pytest.approx(result.constraint_residuals.as_dict())
    assert "restart_summaries" in result.diagnostics
    assert "minimum_eigenvalue_of_stilde_squared" in result.diagnostics


def test_full_time_reversal_unitary_has_expected_shape():
    time_reversal_unitary = build_full_time_reversal_unitary(3)

    assert time_reversal_unitary.shape == (12, 12)
    assert np.allclose(time_reversal_unitary @ time_reversal_unitary.conj().T, np.eye(12))


def test_reference_projector_commutes_with_reference_operator():
    pauli = pauli_matrices()
    orbital_projector = np.array([[1, 0], [0, 0]], dtype=np.complex128)
    local_projector = np.kron(pauli["0"], orbital_projector)

    assert np.allclose(local_projector @ reference_local_operator(), reference_local_operator() @ local_projector)


def _central_finite_difference(
    rho,
    parameters,
    *,
    objective,
    barrier_epsilon=1e-12,
):
    epsilon = 1e-6
    finite_difference = np.zeros_like(parameters)
    for index in range(parameters.size):
        plus = parameters.copy()
        minus = parameters.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        value_plus, _ = evaluate_objective_and_gradient(
            rho,
            plus,
            objective=objective,
            barrier_epsilon=barrier_epsilon,
        )
        value_minus, _ = evaluate_objective_and_gradient(
            rho,
            minus,
            objective=objective,
            barrier_epsilon=barrier_epsilon,
        )
        finite_difference[index] = (value_plus - value_minus) / (2 * epsilon)
    return finite_difference

