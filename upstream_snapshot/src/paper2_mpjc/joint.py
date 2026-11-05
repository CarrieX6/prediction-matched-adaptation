"""Outcome-free core of cross-fitted marginal-preserving joint calibration.

The module deliberately has no diagnosis-label input in the joint fitting API.
For supplied per-concept marginals ``s_j(c_j)``, it constructs

    q_phi(c | s) = IProj_M(s) [ prod_j s_j(c_j) exp(r_phi(c)) ],

where ``M(s)`` contains joint distributions with exactly those one-concept
marginals.  The interaction is the rank-one state-embedding model

    r_phi(c) = sum_{j<k} v_j(c_j) v_k(c_k),

with a zero-sum Helmert parameterization for every ``v_j``.  Consequently,
``phi == 0`` is exactly the independent product.  Joint parameters can be fit
only from concept labels by held-out concept joint likelihood.

All arrays use float64.  The NumPy/SciPy likelihood gradient is analytic: it
uses implicit differentiation of the marginal constraints after IPF
convergence, rather than differentiating through a fixed number of IPF steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Sequence

import numpy as np
from scipy.linalg import helmert
from scipy.optimize import OptimizeResult, minimize


Array = np.ndarray

# One common numerical contract is used by fitting, evaluation, and direct API
# calls.  The 3,000-update hard cap is shared across every continuation stage.
# It was fixed before the D1 hash freeze from outcome-free synthetic stresses in
# the formal seven-concept, 972-state space.  The solver stops as soon as the
# requested marginal tolerance is met and does not change the statistical
# model or its allowed parameter domain.
DEFAULT_IPF_TOLERANCE = 1e-11
DEFAULT_IPF_MAX_ITERATIONS = 3_000

# Deterministic continuation contract.  If the full interaction energy spans
# more than 32 nats, it is introduced in ceil(span / 32) equal increments.
# Intermediate stages solve the same marginal constraints to 1e-8; the final
# stage always uses the caller's requested tolerance.  Each dual Newton update
# is clipped so that no state's unary log scaling changes by more than 2 nats;
# covariance directions below 1e-14 are treated as numerical null directions.
# These constants were fixed from outcome-free synthetic stress only.
CONTINUATION_ENERGY_SPAN = 32.0
CONTINUATION_INTERMEDIATE_TOLERANCE = 1e-8
CONTINUATION_MAX_LOG_STEP = 2.0
CONTINUATION_EIGENVALUE_FLOOR = 1e-14


def _validate_cardinalities(cardinalities: Sequence[int]) -> tuple[int, ...]:
    values = tuple(int(value) for value in cardinalities)
    if len(values) < 2:
        raise ValueError("at least two concepts are required")
    if any(value < 2 for value in values):
        raise ValueError("every concept must have at least two states")
    return values


def enumerate_cartesian_states(cardinalities: Sequence[int]) -> Array:
    """Enumerate concept states in NumPy/C-order (last concept varies fastest)."""

    cards = _validate_cardinalities(cardinalities)
    return np.asarray(list(product(*(range(value) for value in cards))), dtype=np.int64)


def _normalize_marginals(
    marginals: Sequence[Array],
) -> tuple[tuple[Array, ...], tuple[int, ...]]:
    if len(marginals) < 2:
        raise ValueError("at least two marginal arrays are required")

    normalized: list[Array] = []
    batch_size: int | None = None
    for concept_index, marginal in enumerate(marginals):
        value = np.asarray(marginal, dtype=np.float64)
        if value.ndim != 2 or value.shape[1] < 2:
            raise ValueError(
                f"marginal {concept_index} must have shape (n, states>=2)"
            )
        if not np.isfinite(value).all() or np.any(value < 0.0):
            raise ValueError(f"marginal {concept_index} contains invalid probabilities")
        if batch_size is None:
            batch_size = int(value.shape[0])
        elif value.shape[0] != batch_size:
            raise ValueError("marginal batch sizes differ")
        row_sum = value.sum(axis=1, keepdims=True)
        if np.any(row_sum <= 0.0):
            raise ValueError(f"marginal {concept_index} has an empty probability row")
        normalized.append(value / row_sum)

    cards = tuple(int(value.shape[1]) for value in normalized)
    return tuple(normalized), _validate_cardinalities(cards)


def _helmert_basis(cardinality: int) -> Array:
    """Return an orthonormal, zero-column-sum state contrast basis."""

    return np.asarray(helmert(cardinality, full=False).T, dtype=np.float64)


def parameter_dimension(cardinalities: Sequence[int]) -> int:
    """Number of free scalar state-embedding coordinates."""

    cards = _validate_cardinalities(cardinalities)
    return int(sum(value - 1 for value in cards))


def rank1_centered_embeddings(
    phi: Array, cardinalities: Sequence[int]
) -> tuple[Array, ...]:
    """Decode ``phi`` into one centered scalar embedding per concept state."""

    cards = _validate_cardinalities(cardinalities)
    parameter = np.asarray(phi, dtype=np.float64).reshape(-1)
    expected = parameter_dimension(cards)
    if parameter.size != expected:
        raise ValueError(f"expected {expected} parameters, received {parameter.size}")
    if not np.isfinite(parameter).all():
        raise ValueError("phi contains a non-finite value")

    embeddings: list[Array] = []
    offset = 0
    for cardinality in cards:
        width = cardinality - 1
        embeddings.append(_helmert_basis(cardinality) @ parameter[offset : offset + width])
        offset += width
    return tuple(embeddings)


def rank1_interaction(
    phi: Array,
    cardinalities: Sequence[int],
    *,
    states: Array | None = None,
) -> Array:
    """Evaluate ``sum_{j<k} v_j(c_j)v_k(c_k)`` on every Cartesian state."""

    cards = _validate_cardinalities(cardinalities)
    state_table = (
        enumerate_cartesian_states(cards)
        if states is None
        else np.asarray(states, dtype=np.int64)
    )
    _validate_state_table(state_table, cards)
    embeddings = rank1_centered_embeddings(phi, cards)
    embedded_states = np.stack(
        [embedding[state_table[:, j]] for j, embedding in enumerate(embeddings)],
        axis=1,
    )
    summed = embedded_states.sum(axis=1)
    return 0.5 * (np.square(summed) - np.square(embedded_states).sum(axis=1))


def _validate_state_table(states: Array, cardinalities: Sequence[int]) -> None:
    cards = tuple(cardinalities)
    expected_rows = int(np.prod(cards, dtype=np.int64))
    if states.shape != (expected_rows, len(cards)):
        raise ValueError(
            f"states must have shape ({expected_rows}, {len(cards)}), got {states.shape}"
        )
    for j, cardinality in enumerate(cards):
        if np.any(states[:, j] < 0) or np.any(states[:, j] >= cardinality):
            raise ValueError(f"states contain an invalid value for concept {j}")
    if np.unique(states, axis=0).shape[0] != expected_rows:
        raise ValueError("states must enumerate each Cartesian state exactly once")


def independent_product(
    marginals: Sequence[Array], *, states: Array | None = None
) -> Array:
    """Return the independent Cartesian product of normalized marginals."""

    supplied, cards = _normalize_marginals(marginals)
    state_table = (
        enumerate_cartesian_states(cards)
        if states is None
        else np.asarray(states, dtype=np.int64)
    )
    _validate_state_table(state_table, cards)
    return _independent_product_normalized(supplied, state_table)


def _independent_product_normalized(
    marginals: Sequence[Array], states: Array
) -> Array:
    """Product helper for marginals that were already normalized once."""

    joint = np.ones((marginals[0].shape[0], states.shape[0]), dtype=np.float64)
    for j, marginal in enumerate(marginals):
        joint *= marginal[:, states[:, j]]
    return joint


def _state_indicators(states: Array, cardinalities: Sequence[int]) -> tuple[Array, ...]:
    return tuple(
        np.eye(cardinality, dtype=np.float64)[states[:, j]]
        for j, cardinality in enumerate(cardinalities)
    )


def _marginal_error(
    joint: Array, marginals: Sequence[Array], indicators: Sequence[Array]
) -> float:
    return float(
        max(
            np.max(np.abs(joint @ indicator - marginal))
            for indicator, marginal in zip(indicators, marginals)
        )
    )


@dataclass(frozen=True)
class ProjectionDiagnostics:
    """Numerical audit of a multidimensional IPF projection."""

    iterations: int
    max_marginal_error: float
    converged: bool
    stages: int = 1
    stage_iterations: tuple[int, ...] = ()
    energy_span: float = 0.0
    solver: str = "cyclic_ipf"


def _ipf_project(
    kernel: Array,
    marginals: Sequence[Array],
    states: Array,
    *,
    tolerance: float,
    max_iterations: int,
) -> tuple[Array, ProjectionDiagnostics]:
    """I-project a positive kernel onto supplied one-way marginal constraints."""

    if tolerance <= 0.0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be finite and positive")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    joint = np.asarray(kernel, dtype=np.float64).copy()
    if joint.shape != (marginals[0].shape[0], states.shape[0]):
        raise ValueError("kernel shape does not match marginals and state space")
    if not np.isfinite(joint).all() or np.any(joint < 0.0):
        raise ValueError("kernel must be finite and non-negative")
    normalizer = joint.sum(axis=1, keepdims=True)
    if np.any(normalizer <= 0.0):
        raise ValueError("kernel contains an empty row")
    joint /= normalizer

    cards = tuple(value.shape[1] for value in marginals)
    indicators = _state_indicators(states, cards)
    error = _marginal_error(joint, marginals, indicators)
    if error <= tolerance:
        return joint, ProjectionDiagnostics(
            0, error, True, stage_iterations=(0,)
        )

    tiny = np.finfo(np.float64).tiny
    for iteration in range(1, max_iterations + 1):
        for j, (target, indicator) in enumerate(zip(marginals, indicators)):
            current = joint @ indicator
            ratio = np.zeros_like(current)
            positive_target = target > 0.0
            if np.any(positive_target & (current <= 0.0)):
                raise FloatingPointError("IPF encountered an infeasible positive marginal")
            np.divide(
                target,
                np.maximum(current, tiny),
                out=ratio,
                where=positive_target,
            )
            joint *= ratio[:, states[:, j]]
        error = _marginal_error(joint, marginals, indicators)
        if error <= tolerance:
            return joint, ProjectionDiagnostics(
                iteration, error, True, stage_iterations=(iteration,)
            )

    return joint, ProjectionDiagnostics(
        max_iterations, error, False, stage_iterations=(max_iterations,)
    )


def _case_marginal_error(
    joint: Array, marginals: Sequence[Array], indicators: Sequence[Array]
) -> Array:
    """Maximum supplied-marginal error for each batch row."""

    error = np.zeros(joint.shape[0], dtype=np.float64)
    for indicator, marginal in zip(indicators, marginals):
        error = np.maximum(
            error,
            np.max(np.abs(joint @ indicator - marginal), axis=1),
        )
    return error


def _softmax_from_energy(energy: Array) -> Array:
    shifted = energy - np.max(energy, axis=1, keepdims=True)
    probability = np.exp(shifted)
    normalizer = probability.sum(axis=1, keepdims=True)
    if np.any(normalizer <= 0.0) or not np.isfinite(normalizer).all():
        raise FloatingPointError("continuation produced an empty probability row")
    return probability / normalizer


def _continuation_ipf_project(
    log_base: Array,
    interaction: Array,
    marginals: Sequence[Array],
    states: Array,
    *,
    tolerance: float,
    max_iterations: int,
) -> tuple[Array, ProjectionDiagnostics]:
    """Solve a strong-energy I-projection by deterministic continuation.

    Let ``k_t(c)=prod_j s_j(c_j) exp(t r(c))``.  If a previous stage has
    acquired any unary scaling ``exp(u(c))``, then for every ``q`` in the
    supplied-marginal set ``E_q[u]`` is constant.  Consequently

    ``IProj_M(k_t exp(u)) == IProj_M(k_t)``.

    We may therefore warm-start each energy stage with the preceding unary
    scaling without changing the final projection.  Unary scaling is solved in
    reduced Helmert coordinates.  A damped Newton update in these coordinates
    remains an IPF/generalized-iterative-scaling update because it multiplies
    the table only by per-concept state factors.  ``max_iterations`` is one
    explicit hard cap summed across every continuation stage; it is never reset
    or silently extended.
    """

    if tolerance <= 0.0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be finite and positive")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    if log_base.shape != (marginals[0].shape[0], states.shape[0]):
        raise ValueError("log_base shape does not match marginals and states")
    if interaction.shape != (states.shape[0],):
        raise ValueError("interaction shape does not match states")

    cards = tuple(value.shape[1] for value in marginals)
    indicators = _state_indicators(states, cards)
    unary = _unary_contrast_features(cards, states)
    unary_outer = np.einsum("sh,sk->shk", unary, unary, optimize=True).reshape(
        states.shape[0], -1
    )
    target_moment = np.concatenate(
        [
            marginal @ _helmert_basis(cardinality)
            for marginal, cardinality in zip(marginals, cards)
        ],
        axis=1,
    )
    energy_span = float(np.ptp(interaction))
    stages = max(1, int(np.ceil(energy_span / CONTINUATION_ENERGY_SPAN)))
    unary_parameter = np.zeros_like(target_moment)
    total_iterations = 0
    stage_iterations: list[int] = []
    converged = True

    for stage in range(1, stages + 1):
        fraction = float(stage) / float(stages)
        stage_tolerance = (
            tolerance
            if stage == stages
            else max(tolerance, CONTINUATION_INTERMEDIATE_TOLERANCE)
        )
        updates_this_stage = 0
        while True:
            energy = (
                log_base
                + fraction * interaction[None, :]
                + unary_parameter @ unary.T
            )
            joint = _softmax_from_energy(energy)
            case_error = _case_marginal_error(joint, marginals, indicators)
            active = case_error > stage_tolerance
            if not np.any(active):
                break
            if total_iterations >= max_iterations:
                converged = False
                break

            active_joint = joint[active]
            active_mean = active_joint @ unary
            gradient = active_mean - target_moment[active]
            covariance = (active_joint @ unary_outer).reshape(
                active_joint.shape[0], unary.shape[1], unary.shape[1]
            )
            covariance -= np.einsum(
                "nh,nk->nhk", active_mean, active_mean, optimize=True
            )

            eigenvalue, eigenvector = np.linalg.eigh(covariance)
            inverse_eigenvalue = np.zeros_like(eigenvalue)
            usable = eigenvalue > CONTINUATION_EIGENVALUE_FLOOR
            inverse_eigenvalue[usable] = 1.0 / eigenvalue[usable]
            rotated_gradient = np.einsum(
                "nhi,nh->ni", eigenvector, gradient, optimize=True
            )
            delta = -np.einsum(
                "nhi,ni->nh",
                eigenvector,
                inverse_eigenvalue * rotated_gradient,
                optimize=True,
            )
            state_log_delta = delta @ unary.T
            maximum = np.max(np.abs(state_log_delta), axis=1)
            damping = np.minimum(
                1.0,
                CONTINUATION_MAX_LOG_STEP / np.maximum(maximum, 1e-300),
            )
            delta *= damping[:, None]
            if not np.isfinite(delta).all():
                raise FloatingPointError("continuation unary update became non-finite")
            unary_parameter[active] += delta
            total_iterations += 1
            updates_this_stage += 1

        stage_iterations.append(updates_this_stage)
        if not converged:
            break

    # Even on a requested non-raising failure, return a table at the final
    # energy rather than an intermediate-stage distribution.
    final_energy = log_base + interaction[None, :] + unary_parameter @ unary.T
    joint = _softmax_from_energy(final_energy)
    maximum_error = float(
        np.max(_case_marginal_error(joint, marginals, indicators))
    )
    converged = bool(converged and maximum_error <= tolerance)
    return joint, ProjectionDiagnostics(
        iterations=total_iterations,
        max_marginal_error=maximum_error,
        converged=converged,
        stages=stages,
        stage_iterations=tuple(stage_iterations),
        energy_span=energy_span,
        solver="energy_continuation_dual_ipf",
    )


def marginal_preserving_joint(
    marginals: Sequence[Array],
    phi: Array,
    *,
    states: Array | None = None,
    tolerance: float = DEFAULT_IPF_TOLERANCE,
    max_iterations: int = DEFAULT_IPF_MAX_ITERATIONS,
    require_convergence: bool = True,
    return_diagnostics: bool = False,
) -> Array | tuple[Array, ProjectionDiagnostics]:
    """Construct a rank-one joint posterior with supplied marginals exactly fixed.

    The dual I-projection is solved independently for every batch row but
    vectorized over cases.  A literal all-zero ``phi`` takes an explicit
    independent-product branch, which makes the null model an exact nested
    special case rather than an approximate consequence of iterative scaling.
    """

    supplied, cards = _normalize_marginals(marginals)
    state_table = (
        enumerate_cartesian_states(cards)
        if states is None
        else np.asarray(states, dtype=np.int64)
    )
    _validate_state_table(state_table, cards)
    parameter = np.asarray(phi, dtype=np.float64).reshape(-1)
    if parameter.size != parameter_dimension(cards):
        raise ValueError(
            f"expected {parameter_dimension(cards)} parameters, received {parameter.size}"
        )

    independent = _independent_product_normalized(supplied, state_table)
    if np.count_nonzero(parameter) == 0:
        indicators = _state_indicators(state_table, cards)
        diagnostics = ProjectionDiagnostics(
            iterations=0,
            max_marginal_error=_marginal_error(independent, supplied, indicators),
            converged=True,
            stages=0,
            stage_iterations=(),
            energy_span=0.0,
            solver="independent_exact",
        )
        return (independent, diagnostics) if return_diagnostics else independent

    interaction = rank1_interaction(parameter, cards, states=state_table)
    # Form the tilted product in log space.  A case-wise scalar shift has no
    # effect on its I-projection and avoids premature underflow when several
    # supplied marginals are small.
    log_base = np.zeros_like(independent)
    with np.errstate(divide="ignore"):
        for j, marginal in enumerate(supplied):
            log_base += np.log(marginal[:, state_table[:, j]])
    # Every non-null parameter uses the same dual continuation solver.  The
    # energy-span rule controls only the number of homotopy stages; it is not a
    # solver switch.  This avoids a numerical discontinuity at the stage
    # boundary and covers slow cyclic-IPF cases immediately below 32 nats.
    joint, diagnostics = _continuation_ipf_project(
        log_base,
        interaction,
        supplied,
        state_table,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    if require_convergence and not diagnostics.converged:
        raise RuntimeError(
            "IPF did not converge: "
            f"max marginal error={diagnostics.max_marginal_error:.3e} after "
            f"{diagnostics.iterations} iterations"
        )
    return (joint, diagnostics) if return_diagnostics else joint


def _validate_concept_labels(
    labels: Array, cardinalities: Sequence[int], expected_rows: int
) -> Array:
    value = np.asarray(labels)
    cards = tuple(cardinalities)
    if value.shape != (expected_rows, len(cards)):
        raise ValueError(
            f"concept_labels must have shape ({expected_rows}, {len(cards)})"
        )
    if not np.issubdtype(value.dtype, np.integer):
        if not np.all(np.equal(value, np.floor(value))):
            raise ValueError("concept_labels must be integers")
    value = value.astype(np.int64, copy=False)
    for j, cardinality in enumerate(cards):
        if np.any(value[:, j] < 0) or np.any(value[:, j] >= cardinality):
            raise ValueError(f"concept_labels contain an invalid value for concept {j}")
    return value


def _flat_state_indices(
    labels: Array, cardinalities: Sequence[int], states: Array
) -> Array:
    """Map labels to rows of a validated, possibly permuted, state table."""

    cards = tuple(cardinalities)
    canonical_label = np.ravel_multi_index(labels.T, cards, mode="raise", order="C")
    canonical_state = np.ravel_multi_index(states.T, cards, mode="raise", order="C")
    inverse = np.empty(states.shape[0], dtype=np.int64)
    inverse[canonical_state] = np.arange(states.shape[0], dtype=np.int64)
    return inverse[canonical_label]


def concept_joint_nll(
    phi: Array,
    marginals: Sequence[Array],
    concept_labels: Array,
    *,
    l2: float = 0.0,
    states: Array | None = None,
    ipf_tolerance: float = DEFAULT_IPF_TOLERANCE,
    ipf_max_iterations: int = DEFAULT_IPF_MAX_ITERATIONS,
) -> float:
    """Mean joint concept NLL; this objective has no diagnosis-label argument."""

    supplied, cards = _normalize_marginals(marginals)
    labels = _validate_concept_labels(concept_labels, cards, supplied[0].shape[0])
    state_table = (
        enumerate_cartesian_states(cards)
        if states is None
        else np.asarray(states, dtype=np.int64)
    )
    joint = marginal_preserving_joint(
        supplied,
        phi,
        states=state_table,
        tolerance=ipf_tolerance,
        max_iterations=ipf_max_iterations,
    )
    selected = joint[
        np.arange(joint.shape[0]), _flat_state_indices(labels, cards, state_table)
    ]
    nll = -np.log(np.clip(selected, np.finfo(np.float64).tiny, 1.0)).mean()
    parameter = np.asarray(phi, dtype=np.float64).reshape(-1)
    if l2 < 0.0 or not np.isfinite(l2):
        raise ValueError("l2 must be finite and non-negative")
    return float(nll + 0.5 * l2 * np.dot(parameter, parameter))


def _interaction_features(
    phi: Array, cardinalities: Sequence[int], states: Array
) -> Array:
    """Return d r_phi(state) / d phi for every state."""

    cards = tuple(cardinalities)
    embeddings = rank1_centered_embeddings(phi, cards)
    embedded_states = np.stack(
        [embedding[states[:, j]] for j, embedding in enumerate(embeddings)], axis=1
    )
    total = embedded_states.sum(axis=1)
    blocks: list[Array] = []
    for j, cardinality in enumerate(cards):
        basis_on_states = _helmert_basis(cardinality)[states[:, j]]
        blocks.append(basis_on_states * (total - embedded_states[:, j])[:, None])
    return np.concatenate(blocks, axis=1)


def _unary_contrast_features(
    cardinalities: Sequence[int], states: Array
) -> Array:
    return np.concatenate(
        [
            _helmert_basis(cardinality)[states[:, j]]
            for j, cardinality in enumerate(cardinalities)
        ],
        axis=1,
    )


def concept_joint_nll_and_gradient(
    phi: Array,
    marginals: Sequence[Array],
    concept_labels: Array,
    *,
    l2: float = 0.0,
    states: Array | None = None,
    ipf_tolerance: float = DEFAULT_IPF_TOLERANCE,
    ipf_max_iterations: int = DEFAULT_IPF_MAX_ITERATIONS,
) -> tuple[float, Array]:
    """Return concept-only NLL and its implicit I-projection gradient.

    IPF introduces case-specific unary scaling factors.  Their derivatives are
    obtained from the marginal constraints using

    ``d alpha / d phi = -Cov(H,H)^(-1) Cov(H,F)``,

    where ``H`` are reduced unary Helmert contrasts and ``F = d r / d phi``.
    This differentiates the converged I-projection without retaining hundreds
    of IPF iterations in an autodiff graph.
    """

    supplied, cards = _normalize_marginals(marginals)
    labels = _validate_concept_labels(concept_labels, cards, supplied[0].shape[0])
    state_table = (
        enumerate_cartesian_states(cards)
        if states is None
        else np.asarray(states, dtype=np.int64)
    )
    _validate_state_table(state_table, cards)
    parameter = np.asarray(phi, dtype=np.float64).reshape(-1)
    if parameter.size != parameter_dimension(cards):
        raise ValueError(
            f"expected {parameter_dimension(cards)} parameters, received {parameter.size}"
        )
    if l2 < 0.0 or not np.isfinite(l2):
        raise ValueError("l2 must be finite and non-negative")

    joint = marginal_preserving_joint(
        supplied,
        parameter,
        states=state_table,
        tolerance=ipf_tolerance,
        max_iterations=ipf_max_iterations,
    )
    flat_labels = _flat_state_indices(labels, cards, state_table)
    selected = joint[np.arange(joint.shape[0]), flat_labels]
    nll = -np.log(np.clip(selected, np.finfo(np.float64).tiny, 1.0)).mean()

    unary = _unary_contrast_features(cards, state_table)  # states x H
    interaction_feature = _interaction_features(parameter, cards, state_table)  # states x P
    mean_unary = joint @ unary
    mean_interaction_feature = joint @ interaction_feature

    unary_outer = np.einsum("sh,sk->shk", unary, unary, optimize=True)
    unary_interaction_outer = np.einsum(
        "sh,sp->shp", unary, interaction_feature, optimize=True
    )
    covariance_unary = (joint @ unary_outer.reshape(state_table.shape[0], -1)).reshape(
        joint.shape[0], unary.shape[1], unary.shape[1]
    )
    covariance_unary -= np.einsum(
        "nh,nk->nhk", mean_unary, mean_unary, optimize=True
    )
    covariance_cross = (
        joint @ unary_interaction_outer.reshape(state_table.shape[0], -1)
    ).reshape(joint.shape[0], unary.shape[1], interaction_feature.shape[1])
    covariance_cross -= np.einsum(
        "nh,np->nhp", mean_unary, mean_interaction_feature, optimize=True
    )

    # With strictly positive soft marginals this covariance is positive definite.
    # A pseudo-inverse fallback keeps zero-probability boundary cases finite.
    try:
        implicit = np.linalg.solve(covariance_unary, covariance_cross)
    except np.linalg.LinAlgError:
        implicit = np.einsum(
            "nhk,nkp->nhp",
            np.linalg.pinv(covariance_unary, rcond=1e-12),
            covariance_cross,
            optimize=True,
        )

    observed_unary = unary[flat_labels]
    observed_interaction_feature = interaction_feature[flat_labels]
    log_probability_score = (
        observed_interaction_feature
        - mean_interaction_feature
        - np.einsum(
            "nh,nhp->np",
            observed_unary - mean_unary,
            implicit,
            optimize=True,
        )
    )
    gradient = -log_probability_score.mean(axis=0) + l2 * parameter
    objective = float(nll + 0.5 * l2 * np.dot(parameter, parameter))
    if not np.isfinite(objective) or not np.isfinite(gradient).all():
        raise FloatingPointError("joint likelihood or gradient became non-finite")
    return objective, np.asarray(gradient, dtype=np.float64)


def finite_difference_gradient(
    function: Callable[[Array], float], phi: Array, *, step: float = 1e-5
) -> Array:
    """Central finite-difference helper for independent gradient audits."""

    parameter = np.asarray(phi, dtype=np.float64).reshape(-1)
    if step <= 0.0 or not np.isfinite(step):
        raise ValueError("step must be finite and positive")
    gradient = np.empty_like(parameter)
    for index in range(parameter.size):
        delta = np.zeros_like(parameter)
        delta[index] = step
        gradient[index] = (function(parameter + delta) - function(parameter - delta)) / (
            2.0 * step
        )
    return gradient


@dataclass(frozen=True)
class JointFit:
    """Fitted outcome-free rank-one joint calibration model."""

    phi: Array
    cardinalities: tuple[int, ...]
    objective: float
    concept_nll: float
    converged: bool
    iterations: int
    gradient_norm: float
    starts: int

    def predict(
        self,
        marginals: Sequence[Array],
        *,
        tolerance: float = DEFAULT_IPF_TOLERANCE,
        max_iterations: int = DEFAULT_IPF_MAX_ITERATIONS,
    ) -> Array:
        supplied, cards = _normalize_marginals(marginals)
        if cards != self.cardinalities:
            raise ValueError(
                f"expected marginal cardinalities {self.cardinalities}, received {cards}"
            )
        return marginal_preserving_joint(
            supplied,
            self.phi,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )

    def marginalize_task(
        self, marginals: Sequence[Array], p_y_given_c: Array
    ) -> Array:
        return exact_task_marginalization(self.predict(marginals), p_y_given_c)


def _deterministic_starts(dimension: int, count: int, scale: float) -> list[Array]:
    if count < 1:
        raise ValueError("n_starts must be positive")
    starts = [np.zeros(dimension, dtype=np.float64)]
    if count == 1:
        return starts
    starts.append(np.full(dimension, scale, dtype=np.float64))
    if count >= 3:
        starts.append(scale * np.where(np.arange(dimension) % 2 == 0, 1.0, -1.0))
    rng = np.random.default_rng(7319)
    while len(starts) < count:
        starts.append(rng.normal(0.0, scale, size=dimension))
    return starts


def fit_rank1_joint(
    marginals: Sequence[Array],
    concept_labels: Array,
    *,
    l2: float = 0.01,
    max_iterations: int = 150,
    ipf_tolerance: float = DEFAULT_IPF_TOLERANCE,
    ipf_max_iterations: int = DEFAULT_IPF_MAX_ITERATIONS,
    n_starts: int = 4,
    start_scale: float = 0.15,
    parameter_bound: float = 4.0,
) -> JointFit:
    """Fit rank-one interactions using concept labels and no task outcomes.

    The all-zero factorization is a stationary point even under correlated data,
    so deterministic non-zero starts are a mathematical necessity rather than a
    diagnosis-guided model-selection choice.  The best concept-NLL objective is
    selected across a frozen set of starts.
    """

    supplied, cards = _normalize_marginals(marginals)
    labels = _validate_concept_labels(concept_labels, cards, supplied[0].shape[0])
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    if start_scale <= 0.0 or not np.isfinite(start_scale):
        raise ValueError("start_scale must be finite and positive")
    if parameter_bound <= 0.0 or not np.isfinite(parameter_bound):
        raise ValueError("parameter_bound must be finite and positive")
    dimension = parameter_dimension(cards)
    state_table = enumerate_cartesian_states(cards)

    def objective(parameter: Array) -> tuple[float, Array]:
        return concept_joint_nll_and_gradient(
            parameter,
            supplied,
            labels,
            l2=l2,
            states=state_table,
            ipf_tolerance=ipf_tolerance,
            ipf_max_iterations=ipf_max_iterations,
        )

    candidates: list[OptimizeResult] = []
    # Keep the exact independent model as an admissible candidate; optimizing
    # from it alone would fail because rank-one factors have zero gradient there.
    null_value, null_gradient = objective(np.zeros(dimension, dtype=np.float64))
    candidates.append(
        OptimizeResult(
            x=np.zeros(dimension, dtype=np.float64),
            fun=null_value,
            jac=null_gradient,
            success=True,
            nit=0,
        )
    )
    for initial in _deterministic_starts(dimension, n_starts, start_scale)[1:]:
        fitted = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            jac=True,
            bounds=[(-parameter_bound, parameter_bound)] * dimension,
            options={
                "maxiter": int(max_iterations),
                "ftol": 1e-11,
                "gtol": 1e-7,
                "maxls": 30,
            },
        )
        if np.isfinite(fitted.fun) and np.isfinite(fitted.x).all():
            candidates.append(fitted)

    selected = min(candidates, key=lambda item: float(item.fun))
    selected_phi = np.asarray(selected.x, dtype=np.float64)
    selected_concept_nll = concept_joint_nll(
        selected_phi,
        supplied,
        labels,
        l2=0.0,
        states=state_table,
        ipf_tolerance=ipf_tolerance,
        ipf_max_iterations=ipf_max_iterations,
    )
    gradient = np.asarray(selected.jac, dtype=np.float64)
    return JointFit(
        phi=selected_phi,
        cardinalities=cards,
        objective=float(selected.fun),
        concept_nll=float(selected_concept_nll),
        converged=bool(selected.success),
        iterations=int(selected.nit),
        gradient_norm=float(np.linalg.norm(gradient)),
        starts=int(n_starts),
    )


def exact_task_marginalization(joint: Array, p_y_given_c: Array) -> Array:
    """Compute ``p(y|x)=sum_c q(c|x)p(y|c)`` over all enumerated states."""

    joint_probability = np.asarray(joint, dtype=np.float64)
    task_table = np.asarray(p_y_given_c, dtype=np.float64)
    if joint_probability.ndim != 2 or task_table.ndim != 2:
        raise ValueError("joint and p_y_given_c must both be two-dimensional")
    if joint_probability.shape[1] != task_table.shape[0]:
        raise ValueError("p_y_given_c rows must match the number of joint states")
    if task_table.shape[1] < 2:
        raise ValueError("p_y_given_c must contain at least two task classes")
    if (
        not np.isfinite(joint_probability).all()
        or not np.isfinite(task_table).all()
        or np.any(joint_probability < 0.0)
        or np.any(task_table < 0.0)
    ):
        raise ValueError("joint or p_y_given_c contains invalid probabilities")
    joint_sum = joint_probability.sum(axis=1, keepdims=True)
    task_sum = task_table.sum(axis=1, keepdims=True)
    if np.any(joint_sum <= 0.0) or np.any(task_sum <= 0.0):
        raise ValueError("joint and p_y_given_c rows must have positive mass")
    normalized_joint = joint_probability / joint_sum
    normalized_task = task_table / task_sum
    marginalized = normalized_joint @ normalized_task
    return marginalized / marginalized.sum(axis=1, keepdims=True)
