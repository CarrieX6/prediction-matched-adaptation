"""Concept-only calibration for CF-MPJC.

The upstream concept probes are trained with inverse-frequency sample weights.
Their probability scores therefore correspond to an approximately uniform class
prior.  Every score row must first be corrected with the empirical prior of the
*probe-fit rows that generated that score*.  The pure calibration layer then
fits one scalar temperature using concept labels only and never changes a prior.

No diagnosis endpoint is accepted by any API in this module.  Cross-fitting is
explicit: every returned training-row probability is produced by a calibrator
whose fit indices are disjoint from that row.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import minimize_scalar


DEFAULT_EPSILON = 1e-12


def _as_labels(labels: np.ndarray | Sequence[int], n_classes: int) -> np.ndarray:
    raw = np.asarray(labels)
    if raw.ndim != 1:
        raise ValueError("labels must be a one-dimensional array")
    if len(raw) == 0:
        raise ValueError("labels must not be empty")
    if not np.all(np.isfinite(raw)):
        raise ValueError("labels must be finite")
    integer = raw.astype(np.int64)
    if not np.array_equal(raw, integer):
        raise ValueError("labels must contain integer class indices")
    if np.any(integer < 0) or np.any(integer >= n_classes):
        raise ValueError("a label is outside the score class range")
    return integer


def normalize_probability(
    probability: np.ndarray | Sequence[Sequence[float]],
) -> np.ndarray:
    """Return finite, non-negative row-normalized probabilities.

    Inputs may be unnormalized non-negative scores.  An all-zero row is
    rejected rather than silently replaced with an arbitrary distribution.
    """

    output = np.asarray(probability, dtype=np.float64)
    if output.ndim != 2:
        raise ValueError("probability scores must be a two-dimensional array")
    if output.shape[0] == 0 or output.shape[1] == 0:
        raise ValueError("probability scores must have non-zero dimensions")
    if not np.all(np.isfinite(output)):
        raise ValueError("probability scores must be finite")
    if np.any(output < 0.0):
        raise ValueError("probability scores must be non-negative")
    denominator = output.sum(axis=1, keepdims=True)
    if np.any(denominator <= 0.0):
        raise ValueError("each probability-score row must have positive mass")
    output = output / denominator
    # A second normalization removes the small row-sum drift that can be
    # introduced by downstream clipping or multiplication.
    return output / output.sum(axis=1, keepdims=True)


def stable_log_probability(
    probability: np.ndarray | Sequence[Sequence[float]],
    epsilon: float = DEFAULT_EPSILON,
) -> np.ndarray:
    """Compute finite log probabilities without changing the public scores."""

    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must lie strictly between zero and one")
    normalized = normalize_probability(probability)
    return np.log(np.clip(normalized, epsilon, 1.0))


def multiclass_nll(
    probability: np.ndarray | Sequence[Sequence[float]],
    labels: np.ndarray | Sequence[int],
    epsilon: float = DEFAULT_EPSILON,
) -> float:
    """Mean multiclass negative log likelihood with stable zero handling."""

    normalized = normalize_probability(probability)
    target = _as_labels(labels, normalized.shape[1])
    if len(target) != len(normalized):
        raise ValueError("probability scores and labels have different lengths")
    log_probability = stable_log_probability(normalized, epsilon=epsilon)
    return float(-np.mean(log_probability[np.arange(len(target)), target]))


def empirical_class_prior(
    labels: np.ndarray | Sequence[int],
    n_classes: int,
) -> np.ndarray:
    """Unsmoothed empirical class prior of a concept calibration fit fold."""

    if n_classes < 1:
        raise ValueError("n_classes must be positive")
    target = _as_labels(labels, n_classes)
    prior = np.bincount(target, minlength=n_classes).astype(np.float64)
    prior /= float(len(target))
    return prior


def correct_inverse_frequency_prior(
    probability: np.ndarray | Sequence[Sequence[float]],
    empirical_prior: np.ndarray | Sequence[float],
    epsilon: float = DEFAULT_EPSILON,
) -> np.ndarray:
    """Undo an inverse-frequency (uniform-prior) concept-probe score.

    If ``s_k(x)`` is the score learned under inverse-frequency weighting, the
    restored score is proportional to ``s_k(x) * pi_k``, where ``pi_k`` is the
    empirical fit-fold prior.  Zero-prior classes remain at zero.  A tiny floor
    is applied only to the incoming score so that a pathological score assigning
    all mass to unseen classes still has a defined projection onto observed
    classes.
    """

    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must lie strictly between zero and one")
    score = normalize_probability(probability)
    prior = np.asarray(empirical_prior, dtype=np.float64)
    if prior.ndim != 1 or len(prior) != score.shape[1]:
        raise ValueError("empirical_prior must match the score class dimension")
    if not np.all(np.isfinite(prior)) or np.any(prior < 0.0):
        raise ValueError("empirical_prior must be finite and non-negative")
    if prior.sum() <= 0.0:
        raise ValueError("empirical_prior must have positive mass")
    prior = prior / prior.sum()
    corrected = np.maximum(score, epsilon) * prior[None, :]
    return normalize_probability(corrected)


def correct_inverse_frequency_prior_rowwise(
    probability: np.ndarray | Sequence[Sequence[float]],
    probe_fit_prior: np.ndarray | Sequence[Sequence[float]],
    epsilon: float = DEFAULT_EPSILON,
) -> np.ndarray:
    """Undo inverse-frequency weighting with the prior that generated each row.

    Cross-fitted probe scores need not share a prior: row ``i`` must be
    corrected with the empirical class prior of the probe fit-fold that
    generated row ``i``.  ``probe_fit_prior`` therefore has shape ``(n, k)``.
    This operation belongs immediately after probe prediction, before any
    calibration-fold temperature fitting.
    """

    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must lie strictly between zero and one")
    score = normalize_probability(probability)
    prior = np.asarray(probe_fit_prior, dtype=np.float64)
    if prior.shape != score.shape:
        raise ValueError("probe_fit_prior must have one class-prior row per score row")
    prior = normalize_probability(prior)
    corrected = np.maximum(score, epsilon) * prior
    return normalize_probability(corrected)


def _temperature_probability(
    probability: np.ndarray,
    temperature: float,
    epsilon: float,
) -> np.ndarray:
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    base = normalize_probability(probability)
    logits = np.log(np.clip(base, epsilon, 1.0)) / float(temperature)
    logits -= np.max(logits, axis=1, keepdims=True)
    output = np.exp(logits)
    return normalize_probability(output)


class ScalarTemperatureCalibrator:
    """Fit one concept-only temperature to already prior-corrected scores.

    This class deliberately has no prior-estimation or prior-correction step.
    The caller must first correct every probe score with the empirical prior of
    the probe fit-fold that produced it.  Outer-validation scores must likewise
    be corrected with the outer fit-all probe prior before ``transform``.
    """

    def __init__(
        self,
        *,
        epsilon: float = DEFAULT_EPSILON,
        temperature_bounds: tuple[float, float] = (0.05, 20.0),
        optimization_tolerance: float = 1e-8,
    ) -> None:
        if not 0.0 < epsilon < 1.0:
            raise ValueError("epsilon must lie strictly between zero and one")
        lower, upper = (float(value) for value in temperature_bounds)
        if not 0.0 < lower <= 1.0 <= upper:
            raise ValueError("temperature_bounds must be positive and contain 1")
        if optimization_tolerance <= 0.0:
            raise ValueError("optimization_tolerance must be positive")
        self.epsilon = float(epsilon)
        self.temperature_bounds = (lower, upper)
        self.optimization_tolerance = float(optimization_tolerance)

        self.n_classes_: int | None = None
        self.temperature_: float | None = None
        self.n_fit_: int | None = None
        self.nll_at_unit_temperature_: float | None = None
        self.training_nll_: float | None = None
        self.optimization_success_: bool | None = None

    def fit(
        self,
        probability: np.ndarray | Sequence[Sequence[float]],
        labels: np.ndarray | Sequence[int],
    ) -> "ScalarTemperatureCalibrator":
        score = normalize_probability(probability)
        target = _as_labels(labels, score.shape[1])
        if len(target) != len(score):
            raise ValueError("probability scores and labels have different lengths")

        self.n_classes_ = int(score.shape[1])
        self.n_fit_ = int(len(target))

        def objective(log_temperature: float) -> float:
            temperature = float(np.exp(log_temperature))
            calibrated = _temperature_probability(score, temperature, self.epsilon)
            return multiclass_nll(calibrated, target, epsilon=self.epsilon)

        baseline = objective(0.0)
        lower, upper = self.temperature_bounds
        if lower == upper:
            candidate_temperature = lower
            candidate_nll = objective(float(np.log(candidate_temperature)))
            optimization_success = True
        else:
            result = minimize_scalar(
                objective,
                bounds=(float(np.log(lower)), float(np.log(upper))),
                method="bounded",
                options={"xatol": self.optimization_tolerance},
            )
            candidate_temperature = float(np.exp(result.x)) if result.success else 1.0
            candidate_nll = objective(float(np.log(candidate_temperature)))
            optimization_success = bool(result.success)

        if not np.isfinite(candidate_nll) or candidate_nll > baseline:
            self.temperature_ = 1.0
            self.training_nll_ = float(baseline)
            self.optimization_success_ = False
        else:
            self.temperature_ = float(candidate_temperature)
            self.training_nll_ = float(candidate_nll)
            self.optimization_success_ = optimization_success
        self.nll_at_unit_temperature_ = float(baseline)
        return self

    def transform(
        self,
        probability: np.ndarray | Sequence[Sequence[float]],
    ) -> np.ndarray:
        if self.n_classes_ is None or self.temperature_ is None:
            raise RuntimeError("calibrator has not been fit")
        score = normalize_probability(probability)
        if score.shape[1] != self.n_classes_:
            raise ValueError("score class dimension differs from the fitted calibrator")
        return _temperature_probability(score, self.temperature_, self.epsilon)

    def fit_transform(
        self,
        probability: np.ndarray | Sequence[Sequence[float]],
        labels: np.ndarray | Sequence[int],
    ) -> np.ndarray:
        return self.fit(probability, labels).transform(probability)


class PriorTemperatureCalibrator:
    """Empirical-prior correction followed by scalar temperature scaling."""

    def __init__(
        self,
        *,
        epsilon: float = DEFAULT_EPSILON,
        temperature_bounds: tuple[float, float] = (0.05, 20.0),
        optimization_tolerance: float = 1e-8,
    ) -> None:
        if not 0.0 < epsilon < 1.0:
            raise ValueError("epsilon must lie strictly between zero and one")
        lower, upper = (float(value) for value in temperature_bounds)
        if not 0.0 < lower <= 1.0 <= upper:
            raise ValueError("temperature_bounds must be positive and contain 1")
        if optimization_tolerance <= 0.0:
            raise ValueError("optimization_tolerance must be positive")
        self.epsilon = float(epsilon)
        self.temperature_bounds = (lower, upper)
        self.optimization_tolerance = float(optimization_tolerance)

        self.n_classes_: int | None = None
        self.empirical_prior_: np.ndarray | None = None
        self.temperature_: float | None = None
        self.constant_class_: int | None = None
        self.n_fit_: int | None = None
        self.nll_at_unit_temperature_: float | None = None
        self.training_nll_: float | None = None
        self.optimization_success_: bool | None = None

    def fit(
        self,
        probability: np.ndarray | Sequence[Sequence[float]],
        labels: np.ndarray | Sequence[int],
    ) -> "PriorTemperatureCalibrator":
        score = normalize_probability(probability)
        target = _as_labels(labels, score.shape[1])
        if len(target) != len(score):
            raise ValueError("probability scores and labels have different lengths")

        # Clear all learned state so re-fitting a previously constant
        # calibrator cannot retain a stale constant-class shortcut.
        self.constant_class_ = None
        self.temperature_ = None
        self.nll_at_unit_temperature_ = None
        self.training_nll_ = None
        self.optimization_success_ = None
        self.n_classes_ = int(score.shape[1])
        self.n_fit_ = int(len(target))
        self.empirical_prior_ = empirical_class_prior(target, self.n_classes_)
        observed = np.flatnonzero(self.empirical_prior_ > 0.0)

        if len(observed) == 1:
            self.constant_class_ = int(observed[0])
            self.temperature_ = 1.0
            self.nll_at_unit_temperature_ = 0.0
            self.training_nll_ = 0.0
            self.optimization_success_ = True
            return self

        corrected = correct_inverse_frequency_prior(
            score, self.empirical_prior_, epsilon=self.epsilon
        )

        def objective(log_temperature: float) -> float:
            temperature = float(np.exp(log_temperature))
            calibrated = _temperature_probability(corrected, temperature, self.epsilon)
            return multiclass_nll(calibrated, target, epsilon=self.epsilon)

        baseline = objective(0.0)
        lower, upper = self.temperature_bounds
        result = minimize_scalar(
            objective,
            bounds=(float(np.log(lower)), float(np.log(upper))),
            method="bounded",
            options={"xatol": self.optimization_tolerance},
        )
        candidate_temperature = float(np.exp(result.x)) if result.success else 1.0
        candidate_nll = objective(float(np.log(candidate_temperature)))

        # Unit temperature is an explicit candidate.  This makes the claimed
        # in-fit non-inferiority invariant robust to optimizer edge cases.
        if not np.isfinite(candidate_nll) or candidate_nll > baseline:
            self.temperature_ = 1.0
            self.training_nll_ = float(baseline)
            self.optimization_success_ = False
        else:
            self.temperature_ = candidate_temperature
            self.training_nll_ = float(candidate_nll)
            self.optimization_success_ = bool(result.success)
        self.nll_at_unit_temperature_ = float(baseline)
        return self

    def _require_fit(self) -> tuple[int, np.ndarray, float]:
        if (
            self.n_classes_ is None
            or self.empirical_prior_ is None
            or self.temperature_ is None
        ):
            raise RuntimeError("calibrator has not been fit")
        return self.n_classes_, self.empirical_prior_, self.temperature_

    def transform(
        self,
        probability: np.ndarray | Sequence[Sequence[float]],
    ) -> np.ndarray:
        n_classes, prior, temperature = self._require_fit()
        score = normalize_probability(probability)
        if score.shape[1] != n_classes:
            raise ValueError("score class dimension differs from the fitted calibrator")
        if self.constant_class_ is not None:
            output = np.zeros_like(score)
            output[:, self.constant_class_] = 1.0
            return output
        corrected = correct_inverse_frequency_prior(score, prior, epsilon=self.epsilon)
        return _temperature_probability(corrected, temperature, self.epsilon)

    def fit_transform(
        self,
        probability: np.ndarray | Sequence[Sequence[float]],
        labels: np.ndarray | Sequence[int],
    ) -> np.ndarray:
        return self.fit(probability, labels).transform(probability)


@dataclass(frozen=True)
class CalibrationFold:
    """Auditable indices and fitted calibrator for one held-out fold."""

    fit_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    calibrator: PriorTemperatureCalibrator


@dataclass(frozen=True)
class CrossFittedCalibration:
    """OOF calibrated training scores plus an outer-validation calibrator."""

    oof_probability: np.ndarray
    validation_fold: np.ndarray
    folds: tuple[CalibrationFold, ...]
    fit_all_calibrator: PriorTemperatureCalibrator

    def transform_outer(
        self,
        probability: np.ndarray | Sequence[Sequence[float]],
    ) -> np.ndarray:
        """Calibrate outer-validation scores using all outer-training rows."""

        return self.fit_all_calibrator.transform(probability)


def cross_fitted_calibration(
    probability: np.ndarray | Sequence[Sequence[float]],
    labels: np.ndarray | Sequence[int],
    splits: Iterable[tuple[np.ndarray | Sequence[int], np.ndarray | Sequence[int]]],
    *,
    epsilon: float = DEFAULT_EPSILON,
    temperature_bounds: tuple[float, float] = (0.05, 20.0),
    optimization_tolerance: float = 1e-8,
) -> CrossFittedCalibration:
    """Calibrate training scores out of fold and fit an all-row calibrator.

    ``splits`` is deliberately supplied by the caller so the calibration layer
    can reuse a frozen experimental split.  Every row must occur in exactly one
    validation fold.  A fold's fit and validation indices must be disjoint.
    """

    score = normalize_probability(probability)
    target = _as_labels(labels, score.shape[1])
    if len(target) != len(score):
        raise ValueError("probability scores and labels have different lengths")

    n = len(target)
    output = np.full_like(score, np.nan)
    validation_fold = np.full(n, -1, dtype=np.int64)
    census = np.zeros(n, dtype=np.int64)
    fold_records: list[CalibrationFold] = []

    for fold_index, (fit_raw, validation_raw) in enumerate(splits):
        fit_indices = np.asarray(fit_raw, dtype=np.int64)
        validation_indices = np.asarray(validation_raw, dtype=np.int64)
        if fit_indices.ndim != 1 or validation_indices.ndim != 1:
            raise ValueError("split indices must be one-dimensional")
        if len(fit_indices) == 0 or len(validation_indices) == 0:
            raise ValueError("every split must have non-empty fit and validation sets")
        if (
            np.any(fit_indices < 0)
            or np.any(fit_indices >= n)
            or np.any(validation_indices < 0)
            or np.any(validation_indices >= n)
        ):
            raise ValueError("a split index is outside the training row range")
        if len(np.unique(fit_indices)) != len(fit_indices):
            raise ValueError("fit indices are duplicated within a split")
        if len(np.unique(validation_indices)) != len(validation_indices):
            raise ValueError("validation indices are duplicated within a split")
        if np.intersect1d(fit_indices, validation_indices).size:
            raise ValueError("calibration fit and validation indices overlap")

        calibrator = PriorTemperatureCalibrator(
            epsilon=epsilon,
            temperature_bounds=temperature_bounds,
            optimization_tolerance=optimization_tolerance,
        ).fit(score[fit_indices], target[fit_indices])
        output[validation_indices] = calibrator.transform(score[validation_indices])
        census[validation_indices] += 1
        validation_fold[validation_indices] = fold_index
        fold_records.append(
            CalibrationFold(
                fit_indices=tuple(int(value) for value in fit_indices),
                validation_indices=tuple(int(value) for value in validation_indices),
                calibrator=calibrator,
            )
        )

    if not fold_records:
        raise ValueError("splits must contain at least one fold")
    if not np.all(census == 1):
        raise ValueError("every training row must occur in exactly one validation fold")
    if not np.all(np.isfinite(output)):
        raise AssertionError("cross-fitted calibration produced incomplete probabilities")

    fit_all = PriorTemperatureCalibrator(
        epsilon=epsilon,
        temperature_bounds=temperature_bounds,
        optimization_tolerance=optimization_tolerance,
    ).fit(score, target)
    return CrossFittedCalibration(
        oof_probability=normalize_probability(output),
        validation_fold=validation_fold,
        folds=tuple(fold_records),
        fit_all_calibrator=fit_all,
    )


@dataclass(frozen=True)
class TemperatureCalibrationFold:
    """Auditable indices for a pure-temperature held-out fold."""

    fit_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    calibrator: ScalarTemperatureCalibrator


@dataclass(frozen=True)
class CrossFittedTemperatureCalibration:
    """Temperature-only OOF scores and the outer-validation calibrator.

    Both input paths accept scores whose inverse-frequency probe prior has
    already been corrected.  Neither path applies or estimates a class prior.
    """

    oof_probability: np.ndarray
    validation_fold: np.ndarray
    folds: tuple[TemperatureCalibrationFold, ...]
    fit_all_calibrator: ScalarTemperatureCalibrator

    def transform_outer(
        self,
        prior_corrected_probability: np.ndarray | Sequence[Sequence[float]],
    ) -> np.ndarray:
        """Temperature-scale already corrected outer-validation scores."""

        return self.fit_all_calibrator.transform(prior_corrected_probability)


def cross_fitted_temperature_calibration(
    prior_corrected_probability: np.ndarray | Sequence[Sequence[float]],
    labels: np.ndarray | Sequence[int],
    splits: Iterable[tuple[np.ndarray | Sequence[int], np.ndarray | Sequence[int]]],
    *,
    epsilon: float = DEFAULT_EPSILON,
    temperature_bounds: tuple[float, float] = (0.05, 20.0),
    optimization_tolerance: float = 1e-8,
) -> CrossFittedTemperatureCalibration:
    """Cross-fit temperature after row-wise probe-prior correction.

    Each returned training-row score is temperature-scaled by a calibrator that
    did not observe that row's concept label.  The fit-all calibrator is fitted
    only for application to outer-validation scores.  Crucially, this function
    never estimates, restores, or otherwise changes a class prior; it receives
    already prior-corrected scores.
    """

    score = normalize_probability(prior_corrected_probability)
    target = _as_labels(labels, score.shape[1])
    if len(target) != len(score):
        raise ValueError("probability scores and labels have different lengths")

    n = len(target)
    output = np.full_like(score, np.nan)
    validation_fold = np.full(n, -1, dtype=np.int64)
    census = np.zeros(n, dtype=np.int64)
    fold_records: list[TemperatureCalibrationFold] = []

    for fold_index, (fit_raw, validation_raw) in enumerate(splits):
        fit_indices = np.asarray(fit_raw, dtype=np.int64)
        validation_indices = np.asarray(validation_raw, dtype=np.int64)
        if fit_indices.ndim != 1 or validation_indices.ndim != 1:
            raise ValueError("split indices must be one-dimensional")
        if len(fit_indices) == 0 or len(validation_indices) == 0:
            raise ValueError("every split must have non-empty fit and validation sets")
        if (
            np.any(fit_indices < 0)
            or np.any(fit_indices >= n)
            or np.any(validation_indices < 0)
            or np.any(validation_indices >= n)
        ):
            raise ValueError("a split index is outside the training row range")
        if len(np.unique(fit_indices)) != len(fit_indices):
            raise ValueError("fit indices are duplicated within a split")
        if len(np.unique(validation_indices)) != len(validation_indices):
            raise ValueError("validation indices are duplicated within a split")
        if np.intersect1d(fit_indices, validation_indices).size:
            raise ValueError("calibration fit and validation indices overlap")

        calibrator = ScalarTemperatureCalibrator(
            epsilon=epsilon,
            temperature_bounds=temperature_bounds,
            optimization_tolerance=optimization_tolerance,
        ).fit(score[fit_indices], target[fit_indices])
        output[validation_indices] = calibrator.transform(score[validation_indices])
        census[validation_indices] += 1
        validation_fold[validation_indices] = fold_index
        fold_records.append(
            TemperatureCalibrationFold(
                fit_indices=tuple(int(value) for value in fit_indices),
                validation_indices=tuple(int(value) for value in validation_indices),
                calibrator=calibrator,
            )
        )

    if not fold_records:
        raise ValueError("splits must contain at least one fold")
    if not np.all(census == 1):
        raise ValueError("every training row must occur in exactly one validation fold")
    if not np.all(np.isfinite(output)):
        raise AssertionError("cross-fitted calibration produced incomplete probabilities")

    fit_all = ScalarTemperatureCalibrator(
        epsilon=epsilon,
        temperature_bounds=temperature_bounds,
        optimization_tolerance=optimization_tolerance,
    ).fit(score, target)
    return CrossFittedTemperatureCalibration(
        oof_probability=normalize_probability(output),
        validation_fold=validation_fold,
        folds=tuple(fold_records),
        fit_all_calibrator=fit_all,
    )


__all__ = [
    "CalibrationFold",
    "CrossFittedCalibration",
    "CrossFittedTemperatureCalibration",
    "PriorTemperatureCalibrator",
    "ScalarTemperatureCalibrator",
    "TemperatureCalibrationFold",
    "correct_inverse_frequency_prior",
    "correct_inverse_frequency_prior_rowwise",
    "cross_fitted_calibration",
    "cross_fitted_temperature_calibration",
    "empirical_class_prior",
    "multiclass_nll",
    "normalize_probability",
    "stable_log_probability",
]
