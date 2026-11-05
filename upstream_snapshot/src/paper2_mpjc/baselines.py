"""Capacity-matched score-only baselines for CF-MPJC.

The rank-one bilinear classifier deliberately receives exactly the same
calibrated per-concept score blocks as CF-MPJC.  It can learn a shared
second-order interaction without constructing a semantic joint posterior.  A
CF-MPJC gain that disappears against this model is therefore a capacity or
parameterization effect, not evidence for joint posterior calibration.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Sequence

import numpy as np
from scipy.linalg import helmert
from scipy.optimize import minimize


def _seed(base: int, *parts: object) -> int:
    payload = "|".join([str(base), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(sha256(payload).digest()[:4], "little") & 0x7FFFFFFF


def _validate_widths(widths: Sequence[int], dimension: int) -> tuple[int, ...]:
    widths = tuple(int(value) for value in widths)
    if not widths or any(value < 2 for value in widths):
        raise ValueError("every concept must have at least two states")
    if sum(widths) != int(dimension):
        raise ValueError("concept widths do not cover the score dimension")
    return widths


def _normalized_scores(score: np.ndarray) -> np.ndarray:
    score = np.asarray(score, dtype=np.float64)
    if score.ndim != 2 or not np.isfinite(score).all() or np.any(score < 0.0):
        raise ValueError("concept scores must be a finite non-negative matrix")
    denominator = score.sum(axis=1, keepdims=True)
    if np.any(denominator <= 0.0):
        raise ValueError("concept-score rows must carry positive mass")
    return score / denominator


def _contrast_features(
    score: np.ndarray, widths: Sequence[int]
) -> tuple[np.ndarray, tuple[slice, ...]]:
    """Map simplex blocks to their non-redundant Helmert coordinates."""

    score = np.asarray(score, dtype=np.float64)
    widths = _validate_widths(widths, score.shape[1])
    blocks: list[np.ndarray] = []
    reduced_slices: list[slice] = []
    source_start = 0
    reduced_start = 0
    for width in widths:
        block = _normalized_scores(score[:, source_start : source_start + width])
        contrast = helmert(width, full=False).T
        reduced = block @ contrast
        blocks.append(reduced)
        reduced_slices.append(slice(reduced_start, reduced_start + width - 1))
        source_start += width
        reduced_start += width - 1
    return np.concatenate(blocks, axis=1), tuple(reduced_slices)


def _balanced_weight(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.int64)
    if y.ndim != 1 or len(y) == 0 or np.any((y < 0) | (y > 1)):
        raise ValueError("the bilinear D1 baseline requires binary labels")
    count = np.bincount(y, minlength=2).astype(np.float64)
    if np.any(count == 0.0):
        raise ValueError("both diagnosis classes are required")
    return np.asarray([len(y) / (2.0 * count[value]) for value in y])


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    output = np.empty_like(value)
    positive = value >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


@dataclass(frozen=True)
class BilinearFitSummary:
    objective: float
    converged: bool
    iterations: int
    start_index: int
    parameter_count: int


class RankOneBilinearClassifier:
    """Binary logistic head with a shared rank-one cross-concept interaction.

    For reduced score blocks ``r_j`` and learned state embeddings ``a_j``, the
    logit is

    ``b + w^T r + sum_{j<k} (r_j^T a_j)(r_k^T a_k)``.

    Its effective parameter count is ``1 + 2 * sum_j(K_j-1)``, close to a
    rank-one MPJC joint layer plus a linear gold-concept diagnosis head.
    """

    def __init__(
        self,
        widths: Sequence[int],
        *,
        l2: float = 0.01,
        starts: int = 5,
        seed: int = 2026090827,
        max_iter: int = 1000,
    ) -> None:
        if l2 < 0.0 or starts < 1 or max_iter < 1:
            raise ValueError("invalid bilinear optimization configuration")
        self.widths = tuple(int(value) for value in widths)
        self.l2 = float(l2)
        self.starts = int(starts)
        self.seed = int(seed)
        self.max_iter = int(max_iter)
        self.parameter_: np.ndarray | None = None
        self.reduced_slices_: tuple[slice, ...] | None = None
        self.summary_: BilinearFitSummary | None = None

    def _unpack(self, parameter: np.ndarray, dimension: int) -> tuple[float, np.ndarray, np.ndarray]:
        parameter = np.asarray(parameter, dtype=np.float64)
        if parameter.shape != (1 + 2 * dimension,):
            raise ValueError("bilinear parameter shape changed")
        return float(parameter[0]), parameter[1 : 1 + dimension], parameter[1 + dimension :]

    def fit(self, score: np.ndarray, y: np.ndarray) -> "RankOneBilinearClassifier":
        score = np.asarray(score, dtype=np.float64)
        reduced, slices = _contrast_features(score, self.widths)
        target = np.asarray(y, dtype=np.int64)
        if len(target) != len(reduced):
            raise ValueError("scores and labels have different lengths")
        sample_weight = _balanced_weight(target)
        dimension = reduced.shape[1]

        def value_and_gradient(parameter: np.ndarray) -> tuple[float, np.ndarray]:
            intercept, linear, latent = self._unpack(parameter, dimension)
            node = np.column_stack(
                [reduced[:, block] @ latent[block] for block in slices]
            )
            node_sum = node.sum(axis=1)
            interaction = 0.5 * (np.square(node_sum) - np.square(node).sum(axis=1))
            logit = intercept + reduced @ linear + interaction
            # log(1 + exp(z)) - y*z is stable through logaddexp.
            data_loss = np.mean(sample_weight * (np.logaddexp(0.0, logit) - target * logit))
            penalty = 0.5 * self.l2 * (np.square(linear).sum() + np.square(latent).sum())
            residual = sample_weight * (_sigmoid(logit) - target) / len(target)
            gradient = np.empty_like(parameter)
            gradient[0] = residual.sum()
            gradient[1 : 1 + dimension] = reduced.T @ residual + self.l2 * linear
            latent_gradient = np.zeros(dimension, dtype=np.float64)
            for node_index, block in enumerate(slices):
                multiplier = residual * (node_sum - node[:, node_index])
                latent_gradient[block] = reduced[:, block].T @ multiplier
            latent_gradient += self.l2 * latent
            gradient[1 + dimension :] = latent_gradient
            return float(data_loss + penalty), gradient

        candidates = []
        for start_index in range(self.starts):
            rng = np.random.default_rng(_seed(self.seed, "bilinear", start_index))
            initial = np.zeros(1 + 2 * dimension, dtype=np.float64)
            # Zero latent factors are a stationary point.  Fixed small nonzero
            # starts make the rank-one alternative genuinely testable.
            initial[1 + dimension :] = rng.normal(0.0, 0.05, size=dimension)
            fitted = minimize(
                value_and_gradient,
                initial,
                method="L-BFGS-B",
                jac=True,
                options={"maxiter": self.max_iter, "ftol": 1e-11, "gtol": 1e-7},
            )
            candidates.append((float(fitted.fun), start_index, fitted))
        objective, start_index, fitted = min(candidates, key=lambda item: item[0])
        if not np.isfinite(objective) or not np.isfinite(fitted.x).all():
            raise RuntimeError("bilinear optimization produced a non-finite fit")
        self.parameter_ = np.asarray(fitted.x, dtype=np.float64)
        self.reduced_slices_ = slices
        self.summary_ = BilinearFitSummary(
            objective=objective,
            converged=bool(fitted.success),
            iterations=int(fitted.nit),
            start_index=int(start_index),
            parameter_count=int(len(self.parameter_)),
        )
        return self

    def decision_function(self, score: np.ndarray) -> np.ndarray:
        if self.parameter_ is None or self.reduced_slices_ is None:
            raise RuntimeError("bilinear classifier has not been fit")
        reduced, slices = _contrast_features(np.asarray(score, dtype=np.float64), self.widths)
        if slices != self.reduced_slices_:
            raise ValueError("concept widths changed after fit")
        intercept, linear, latent = self._unpack(self.parameter_, reduced.shape[1])
        node = np.column_stack([reduced[:, block] @ latent[block] for block in slices])
        interaction = 0.5 * (np.square(node.sum(axis=1)) - np.square(node).sum(axis=1))
        return intercept + reduced @ linear + interaction

    def predict_proba(self, score: np.ndarray) -> np.ndarray:
        positive = _sigmoid(self.decision_function(score))
        return np.column_stack([1.0 - positive, positive])


__all__ = ["BilinearFitSummary", "RankOneBilinearClassifier"]
