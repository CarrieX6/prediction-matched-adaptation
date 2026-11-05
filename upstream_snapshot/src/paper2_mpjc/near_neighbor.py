"""Capacity-matched unprojected correlated-posterior baseline for CF-MPJC.

This baseline uses the same rank-one interaction and calibrated marginal inputs
as CF-MPJC, but it normalizes the tilted independent product directly instead
of I-projecting it back onto the supplied one-concept marginals.  It is thus a
small energy-based correlated posterior and a direct ablation of marginal
preservation.  Fitting uses concept labels only and never accepts task labels.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.linalg import helmert
from scipy.optimize import OptimizeResult, minimize

from .joint import (
    enumerate_cartesian_states,
    independent_product,
    parameter_dimension,
    rank1_centered_embeddings,
    rank1_interaction,
)


Array = np.ndarray


def _validate_inputs(
    marginals: Sequence[Array], concept_labels: Array | None = None
) -> tuple[tuple[Array, ...], tuple[int, ...], Array | None]:
    if len(marginals) < 2:
        raise ValueError("at least two marginal arrays are required")
    normalized: list[Array] = []
    n: int | None = None
    for index, raw in enumerate(marginals):
        value = np.asarray(raw, dtype=np.float64)
        if value.ndim != 2 or value.shape[1] < 2:
            raise ValueError(f"marginal {index} must have shape (n, states>=2)")
        if not np.isfinite(value).all() or np.any(value < 0.0):
            raise ValueError(f"marginal {index} contains invalid values")
        if n is None:
            n = int(value.shape[0])
        elif len(value) != n:
            raise ValueError("marginal batch sizes differ")
        total = value.sum(axis=1, keepdims=True)
        if np.any(total <= 0.0):
            raise ValueError(f"marginal {index} has an empty row")
        normalized.append(value / total)
    cards = tuple(int(value.shape[1]) for value in normalized)
    labels: Array | None = None
    if concept_labels is not None:
        labels = np.asarray(concept_labels)
        if labels.shape != (int(n), len(cards)):
            raise ValueError("concept_labels shape differs from the marginals")
        if not np.issubdtype(labels.dtype, np.integer):
            if not np.all(np.equal(labels, np.floor(labels))):
                raise ValueError("concept_labels must be integers")
        labels = labels.astype(np.int64, copy=False)
        for concept, card in enumerate(cards):
            if np.any(labels[:, concept] < 0) or np.any(labels[:, concept] >= card):
                raise ValueError(f"invalid concept label in column {concept}")
    return tuple(normalized), cards, labels


def _interaction_features(phi: Array, cards: Sequence[int], states: Array) -> Array:
    """Evaluate the derivative of the shared rank-one energy at every state."""

    embeddings = rank1_centered_embeddings(phi, cards)
    embedded = np.stack(
        [embedding[states[:, j]] for j, embedding in enumerate(embeddings)], axis=1
    )
    total = embedded.sum(axis=1)
    blocks = []
    for j, card in enumerate(cards):
        basis = np.asarray(helmert(card, full=False).T, dtype=np.float64)
        blocks.append(basis[states[:, j]] * (total - embedded[:, j])[:, None])
    return np.concatenate(blocks, axis=1)


def unprojected_rank1_joint(
    marginals: Sequence[Array], phi: Array
) -> Array:
    """Normalize ``prod_j s_j(c_j) exp(r_phi(c))`` without I-projection."""

    supplied, cards, _ = _validate_inputs(marginals)
    parameter = np.asarray(phi, dtype=np.float64).reshape(-1)
    if parameter.shape != (parameter_dimension(cards),):
        raise ValueError("phi dimension differs from the concept cardinalities")
    states = enumerate_cartesian_states(cards)
    independent = independent_product(supplied, states=states)
    interaction = rank1_interaction(parameter, cards, states=states)
    with np.errstate(divide="ignore"):
        log_kernel = np.log(independent) + interaction[None, :]
    maximum = np.max(log_kernel, axis=1, keepdims=True)
    kernel = np.exp(log_kernel - maximum)
    total = kernel.sum(axis=1, keepdims=True)
    if not np.isfinite(kernel).all() or np.any(total <= 0.0):
        raise FloatingPointError("unprojected energy normalization failed")
    return kernel / total


def unprojected_concept_nll_and_gradient(
    phi: Array,
    marginals: Sequence[Array],
    concept_labels: Array,
    *,
    l2: float = 0.01,
) -> tuple[float, Array]:
    """Concept-only penalized likelihood and analytic gradient."""

    supplied, cards, labels = _validate_inputs(marginals, concept_labels)
    assert labels is not None
    parameter = np.asarray(phi, dtype=np.float64).reshape(-1)
    if parameter.shape != (parameter_dimension(cards),):
        raise ValueError("phi dimension differs from the concept cardinalities")
    if not np.isfinite(parameter).all() or l2 < 0.0 or not np.isfinite(l2):
        raise ValueError("invalid parameter or regularization")
    states = enumerate_cartesian_states(cards)
    joint = unprojected_rank1_joint(supplied, parameter)
    flat = np.ravel_multi_index(labels.T, cards, mode="raise", order="C")
    selected = joint[np.arange(len(labels)), flat]
    nll = -np.log(np.clip(selected, np.finfo(np.float64).tiny, 1.0)).mean()
    feature = _interaction_features(parameter, cards, states)
    gradient = -(feature[flat] - joint @ feature).mean(axis=0) + l2 * parameter
    objective = float(nll + 0.5 * l2 * np.dot(parameter, parameter))
    if not np.isfinite(objective) or not np.isfinite(gradient).all():
        raise FloatingPointError("unprojected likelihood became non-finite")
    return objective, np.asarray(gradient, dtype=np.float64)


@dataclass(frozen=True)
class UnprojectedJointFit:
    phi: Array
    cardinalities: tuple[int, ...]
    objective: float
    concept_nll: float
    converged: bool
    iterations: int
    gradient_norm: float
    starts: int

    def predict(self, marginals: Sequence[Array]) -> Array:
        _, cards, _ = _validate_inputs(marginals)
        if cards != self.cardinalities:
            raise ValueError("marginal cardinalities differ from the fitted model")
        return unprojected_rank1_joint(marginals, self.phi)


def _starts(dimension: int, count: int, scale: float) -> list[Array]:
    values = [np.zeros(dimension, dtype=np.float64)]
    if count > 1:
        values.append(np.full(dimension, scale, dtype=np.float64))
    if count > 2:
        values.append(scale * np.where(np.arange(dimension) % 2 == 0, 1.0, -1.0))
    rng = np.random.default_rng(7319)
    while len(values) < count:
        values.append(rng.normal(0.0, scale, size=dimension))
    return values


def fit_unprojected_rank1_joint(
    marginals: Sequence[Array],
    concept_labels: Array,
    *,
    l2: float = 0.01,
    max_iterations: int = 150,
    n_starts: int = 4,
    start_scale: float = 0.15,
    parameter_bound: float = 4.0,
) -> UnprojectedJointFit:
    """Fit the energy-posterior near-neighbor using concept labels only."""

    supplied, cards, labels = _validate_inputs(marginals, concept_labels)
    assert labels is not None
    if max_iterations < 1 or n_starts < 1:
        raise ValueError("optimization counts must be positive")
    if start_scale <= 0.0 or parameter_bound <= 0.0:
        raise ValueError("start scale and bounds must be positive")
    dimension = parameter_dimension(cards)

    def objective(value: Array) -> tuple[float, Array]:
        return unprojected_concept_nll_and_gradient(
            value, supplied, labels, l2=l2
        )

    candidates: list[OptimizeResult] = []
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
    for initial in _starts(dimension, n_starts, start_scale)[1:]:
        result = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            jac=True,
            bounds=[(-parameter_bound, parameter_bound)] * dimension,
            options={"maxiter": max_iterations, "ftol": 1e-11, "gtol": 1e-7, "maxls": 30},
        )
        if np.isfinite(result.fun) and np.isfinite(result.x).all():
            candidates.append(result)
    selected = min(candidates, key=lambda item: float(item.fun))
    phi = np.asarray(selected.x, dtype=np.float64)
    concept_nll, _ = unprojected_concept_nll_and_gradient(
        phi, supplied, labels, l2=0.0
    )
    return UnprojectedJointFit(
        phi=phi,
        cardinalities=cards,
        objective=float(selected.fun),
        concept_nll=float(concept_nll),
        converged=bool(selected.success),
        iterations=int(selected.nit),
        gradient_norm=float(np.linalg.norm(np.asarray(selected.jac, dtype=np.float64))),
        starts=int(n_starts),
    )


__all__ = [
    "UnprojectedJointFit",
    "fit_unprojected_rank1_joint",
    "unprojected_concept_nll_and_gradient",
    "unprojected_rank1_joint",
]
