"""Strict nested, development-only pilot for CF-MPJC D1.

The splitters in this module accept case counts, never diagnosis or concept
labels.  Within every outer split, a strict meta split constructs the concept
posterior for each held-out training block.  All probes, probe-fit priors,
inner-OOF temperature inputs, and temperature labels used to create that block
exclude the whole block.  Outer-validation scores similarly use only
outer-training concept information.  The marginal-preserving and unprojected
joint models are then fitted only against the joint concept labels.  Diagnosis
labels enter only the downstream task heads after the posterior has been frozen.

This module never loads a dataset and has no official-test path.  The CLI
adapter is responsible for loading the quarantined train+valid cohort and for
calling :func:`validate_official_test_exclusion` before this pilot can run.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from paper2_evidence_handoff.data import (
    EXPECTED_DEVELOPMENT,
    EXPECTED_EXCLUSIONS,
    EXPECTED_TEST_AFTER_EXCLUSION,
    EvidenceCohort,
)

from .baselines import RankOneBilinearClassifier
from .calibration import (
    ScalarTemperatureCalibrator,
    correct_inverse_frequency_prior_rowwise,
    empirical_class_prior,
)
from .joint import (
    enumerate_cartesian_states,
    exact_task_marginalization,
    independent_product,
    marginal_preserving_joint,
    fit_rank1_joint,
)
from .near_neighbor import fit_unprojected_rank1_joint


MODEL_NAMES = (
    "CF_MPJC",
    "INDEPENDENT_MARGINALIZATION",
    "UNPROJECTED_RANK1_JOINT",
    "JOINT_MAP",
    "MARGINAL_MEAN_PLUGIN",
    "OOF_SOFT_LOGISTIC",
    "OOF_HARD_LOGISTIC",
    "BILINEAR_SCORE_HEAD",
    "GOLD_TO_HARD_PRED",
    "GOLD_ORACLE",
    "DIRECT_X",
)


@dataclass(frozen=True)
class D1Config:
    """All model and split choices that must be frozen before a real D1 run."""

    seed: int = 2026090829
    outer_folds: int = 5
    inner_folds: int = 4
    calibration_folds: int = 4
    repetitions: int = 5
    probe_c: float = 1.0
    diagnosis_c: float = 1.0
    max_iter: int = 2000
    temperature_lower: float = 0.05
    temperature_upper: float = 20.0
    temperature_optimization_tolerance: float = 1e-8
    probability_epsilon: float = 1e-12
    joint_l2: float = 0.01
    joint_max_iterations: int = 150
    joint_n_starts: int = 4
    joint_start_scale: float = 0.15
    joint_parameter_bound: float = 4.0
    ipf_tolerance: float = 1e-11
    ipf_max_iterations: int = 3000
    bilinear_l2: float = 0.01
    bilinear_starts: int = 5
    bilinear_seed: int = 2026090827
    bilinear_max_iter: int = 1000

    def validate(self, n_cases: int) -> None:
        if self.repetitions < 1:
            raise ValueError("repetitions must be positive")
        for name, value in (
            ("outer_folds", self.outer_folds),
            ("inner_folds", self.inner_folds),
            ("calibration_folds", self.calibration_folds),
        ):
            if value < 2:
                raise ValueError(f"{name} must be at least two")
        smallest_outer_fit = n_cases - int(np.ceil(n_cases / self.outer_folds))
        if n_cases < self.outer_folds:
            raise ValueError("outer_folds exceeds the number of cases")
        if smallest_outer_fit < max(self.inner_folds, self.calibration_folds):
            raise ValueError("an inner fold count exceeds the smallest outer fit")
        if self.probe_c <= 0.0 or self.diagnosis_c <= 0.0:
            raise ValueError("logistic regularization constants must be positive")
        if self.max_iter < 1:
            raise ValueError("max_iter must be positive")
        if not 0.0 < self.temperature_lower <= 1.0 <= self.temperature_upper:
            raise ValueError("temperature bounds must be positive and contain one")
        if self.temperature_optimization_tolerance <= 0.0:
            raise ValueError("temperature optimization tolerance must be positive")
        if not 0.0 < self.probability_epsilon < 1.0:
            raise ValueError("probability epsilon must lie in (0,1)")
        if self.joint_l2 < 0.0 or self.bilinear_l2 < 0.0:
            raise ValueError("regularization strengths must be non-negative")
        if (
            min(
                self.joint_max_iterations,
                self.joint_n_starts,
                self.ipf_max_iterations,
                self.bilinear_starts,
                self.bilinear_max_iter,
            )
            < 1
        ):
            raise ValueError("optimization counts must be positive")
        if self.joint_start_scale <= 0.0 or self.joint_parameter_bound <= 0.0:
            raise ValueError("joint start scale and parameter bound must be positive")
        if self.ipf_tolerance <= 0.0:
            raise ValueError("IPF tolerance must be positive")


@dataclass(frozen=True)
class NestedConceptScores:
    """Auditable outputs of the probe-prior-temperature nested stack."""

    train_probability: np.ndarray
    validation_probability: np.ndarray
    train_raw_probability: np.ndarray
    validation_raw_probability: np.ndarray
    train_probe_fit_prior: np.ndarray
    validation_probe_fit_prior: np.ndarray
    train_meta_fold: np.ndarray
    train_probe_fold: np.ndarray
    train_calibration_fold: np.ndarray
    validation_temperature: tuple[float, ...]
    probability_epsilon: float
    inner_manifests: tuple[dict[str, object], ...]
    calibration_manifests: tuple[dict[str, object], ...]


def _seed(base: int, *parts: object) -> int:
    payload = "|".join([str(base), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(sha256(payload).digest()[:4], "little") & 0x7FFFFFFF


def _case_hash(case_id: Sequence[object]) -> str:
    payload = "\n".join(sorted(str(value) for value in case_id)).encode("utf-8")
    return sha256(payload).hexdigest()


def case_kfold_splits(
    n_cases: int, n_splits: int, seed: int
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Return shuffled K-fold indices without accepting any endpoint labels."""

    if n_cases < n_splits or n_splits < 2:
        raise ValueError("invalid case-only K-fold dimensions")
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    return tuple(
        (np.asarray(fit, dtype=np.int64), np.asarray(validation, dtype=np.int64))
        for fit, validation in splitter.split(np.arange(n_cases, dtype=np.int64))
    )


def _validate_partition(
    splits: Sequence[tuple[np.ndarray, np.ndarray]], n_cases: int
) -> None:
    census = np.zeros(n_cases, dtype=np.int64)
    for fit, validation in splits:
        if np.intersect1d(fit, validation).size:
            raise AssertionError("fit and validation cases overlap")
        if len(np.unique(fit)) != len(fit) or len(np.unique(validation)) != len(
            validation
        ):
            raise AssertionError("a split contains duplicate cases")
        census[validation] += 1
    if not np.all(census == 1):
        raise AssertionError("validation folds do not partition the cases exactly once")


def validate_official_test_exclusion(
    cohort: EvidenceCohort,
    official_meta_directory: str | Path,
    *,
    exclusions: Sequence[int] = EXPECTED_EXCLUSIONS,
    expected_development: int | None = EXPECTED_DEVELOPMENT,
    expected_test_after_exclusion: int | None = EXPECTED_TEST_AFTER_EXCLUSION,
) -> dict[str, object]:
    """Prove that a cohort is exactly eligible official train+valid, never test.

    Only the three public index files are read.  No official-test image,
    embedding, diagnosis, or concept field is opened by this check.
    """

    cohort.validate()
    directory = Path(official_meta_directory)
    split_index: dict[str, np.ndarray] = {}
    for split in ("train", "valid", "test"):
        path = directory / f"{split}_indexes.csv"
        table = pd.read_csv(path)
        if list(table.columns) != ["indexes"]:
            raise ValueError(f"official {split} index schema changed")
        values = table["indexes"].to_numpy(dtype=np.int64)
        if len(np.unique(values)) != len(values):
            raise ValueError(f"official {split} indices are duplicated")
        split_index[split] = values
    if any(
        set(split_index[left]).intersection(split_index[right])
        for left, right in (("train", "valid"), ("train", "test"), ("valid", "test"))
    ):
        raise ValueError("official partitions overlap")

    excluded = {int(value) for value in exclusions}
    eligible_development = np.asarray(
        [
            int(value)
            for split in ("train", "valid")
            for value in split_index[split]
            if int(value) not in excluded
        ],
        dtype=np.int64,
    )
    eligible_test = {
        int(value) for value in split_index["test"] if int(value) not in excluded
    }
    observed = np.asarray(cohort.metadata_index, dtype=np.int64)
    if len(np.unique(observed)) != len(observed):
        raise ValueError("cohort metadata indices are duplicated")
    overlap = sorted(set(observed).intersection(eligible_test))
    if overlap:
        raise ValueError(f"official test entered the D1 cohort: {overlap[:10]}")
    if set(observed) != set(eligible_development) or len(observed) != len(
        eligible_development
    ):
        raise ValueError("D1 cohort is not exactly eligible official train+valid")
    if expected_development is not None and len(observed) != int(expected_development):
        raise ValueError("D1 development census changed")
    if expected_test_after_exclusion is not None and len(eligible_test) != int(
        expected_test_after_exclusion
    ):
        raise ValueError("eligible official-test census changed")
    return {
        "included_partitions": ["train", "valid"],
        "development_count": int(len(observed)),
        "official_test_count_after_exclusion": int(len(eligible_test)),
        "official_test_overlap_count": 0,
        "official_test_outcomes_opened": False,
        "development_metadata_index_sha256": sha256(observed.tobytes()).hexdigest(),
        "official_test_index_sha256": sha256(
            np.asarray(sorted(eligible_test), dtype=np.int64).tobytes()
        ).hexdigest(),
    }


class _BalancedClassifier:
    """Fixed L2 logistic head with explicit absent-class probability columns."""

    def __init__(self, n_classes: int, c: float, max_iter: int, seed: int) -> None:
        self.n_classes = int(n_classes)
        self.c = float(c)
        self.max_iter = int(max_iter)
        self.seed = int(seed)
        self.constant_: int | None = None
        self.model_: LogisticRegression | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "_BalancedClassifier":
        feature = np.asarray(x, dtype=np.float64)
        target = np.asarray(y, dtype=np.int64)
        if feature.ndim != 2 or len(feature) != len(target) or len(target) == 0:
            raise ValueError("classifier training arrays are malformed")
        if not np.isfinite(feature).all():
            raise ValueError("classifier features contain non-finite values")
        if np.any(target < 0) or np.any(target >= self.n_classes):
            raise ValueError("classifier label is outside the declared class range")
        observed = np.unique(target)
        if len(observed) == 1:
            self.constant_ = int(observed[0])
            return self
        counts = np.bincount(target, minlength=self.n_classes).astype(np.float64)
        weight = np.asarray(
            [len(target) / (len(observed) * counts[int(value)]) for value in target],
            dtype=np.float64,
        )
        self.model_ = LogisticRegression(
            C=self.c,
            solver="lbfgs",
            multi_class="auto",
            max_iter=self.max_iter,
            random_state=self.seed,
        )
        self.model_.fit(feature, target, sample_weight=weight)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        feature = np.asarray(x, dtype=np.float64)
        if feature.ndim != 2 or not np.isfinite(feature).all():
            raise ValueError("classifier prediction features are malformed")
        output = np.zeros((len(feature), self.n_classes), dtype=np.float64)
        if self.constant_ is not None:
            output[:, self.constant_] = 1.0
        elif self.model_ is not None:
            partial = self.model_.predict_proba(feature)
            output[:, self.model_.classes_.astype(np.int64)] = partial
        else:
            raise RuntimeError("classifier has not been fit")
        return _normalize_task_probability(output)


def _normalize_task_probability(probability: np.ndarray) -> np.ndarray:
    value = np.asarray(probability, dtype=np.float64)
    if value.ndim != 2 or not np.isfinite(value).all() or np.any(value < 0.0):
        raise ValueError("task probability is malformed")
    denominator = value.sum(axis=1, keepdims=True)
    if np.any(denominator <= 0.0):
        raise ValueError("task probability row has no mass")
    return value / denominator


def _concept_layout(cohort: EvidenceCohort) -> tuple[tuple[str, ...], tuple[int, ...]]:
    names = tuple(cohort.concept_states)
    widths = tuple(len(cohort.concept_states[name]) for name in names)
    if len(names) < 2 or any(width < 2 for width in widths):
        raise ValueError("CF-MPJC requires at least two non-degenerate concepts")
    return names, widths


def _block_slices(widths: Sequence[int]) -> tuple[slice, ...]:
    output: list[slice] = []
    start = 0
    for width in widths:
        output.append(slice(start, start + int(width)))
        start += int(width)
    return tuple(output)


def _concept_matrix(cohort: EvidenceCohort, rows: np.ndarray) -> np.ndarray:
    names, _ = _concept_layout(cohort)
    return np.column_stack(
        [np.asarray(cohort.concepts[name], dtype=np.int64)[rows] for name in names]
    )


def _onehot_concepts(labels: np.ndarray, widths: Sequence[int]) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 2 or labels.shape[1] != len(widths):
        raise ValueError("concept label matrix and widths differ")
    blocks = []
    for concept, width in enumerate(widths):
        block = np.zeros((len(labels), int(width)), dtype=np.float64)
        if np.any(labels[:, concept] < 0) or np.any(labels[:, concept] >= width):
            raise ValueError("concept label is out of range")
        block[np.arange(len(labels)), labels[:, concept]] = 1.0
        blocks.append(block)
    return np.concatenate(blocks, axis=1)


def _hard_concepts(probability: np.ndarray, widths: Sequence[int]) -> np.ndarray:
    value = np.asarray(probability, dtype=np.float64)
    blocks = []
    for block_slice in _block_slices(widths):
        source = value[:, block_slice]
        block = np.zeros_like(source)
        block[np.arange(len(source)), np.argmax(source, axis=1)] = 1.0
        blocks.append(block)
    if sum(widths) != value.shape[1]:
        raise ValueError("concept widths do not cover the score matrix")
    return np.concatenate(blocks, axis=1)


def _marginal_blocks(
    probability: np.ndarray, widths: Sequence[int]
) -> tuple[np.ndarray, ...]:
    value = np.asarray(probability, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != sum(widths):
        raise ValueError("concept probability dimension changed")
    output = []
    for block_slice in _block_slices(widths):
        block = value[:, block_slice]
        denominator = block.sum(axis=1, keepdims=True)
        if (
            not np.isfinite(block).all()
            or np.any(block < 0.0)
            or np.any(denominator <= 0.0)
        ):
            raise ValueError("a concept probability block is invalid")
        output.append(block / denominator)
    return tuple(output)


def _fit_probe_blocks(
    cohort: EvidenceCohort,
    fit_rows: np.ndarray,
    predict_rows: np.ndarray,
    config: D1Config,
    seed_parts: tuple[object, ...],
) -> tuple[np.ndarray, np.ndarray]:
    names, widths = _concept_layout(cohort)
    score_blocks: list[np.ndarray] = []
    prior_blocks: list[np.ndarray] = []
    for concept, (name, width) in enumerate(zip(names, widths, strict=True)):
        labels = np.asarray(cohort.concepts[name], dtype=np.int64)
        classifier = _BalancedClassifier(
            width,
            config.probe_c,
            config.max_iter,
            _seed(config.seed, *seed_parts, concept, name),
        ).fit(cohort.x[fit_rows], labels[fit_rows])
        score_blocks.append(classifier.predict_proba(cohort.x[predict_rows]))
        prior = empirical_class_prior(labels[fit_rows], width)
        prior_blocks.append(np.repeat(prior[None, :], len(predict_rows), axis=0))
    return np.concatenate(score_blocks, axis=1), np.concatenate(prior_blocks, axis=1)


def _prior_correct_blocks(
    raw_probability: np.ndarray,
    probe_fit_prior: np.ndarray,
    widths: Sequence[int],
    *,
    epsilon: float = 1e-12,
) -> np.ndarray:
    corrected = []
    for block_slice in _block_slices(widths):
        corrected.append(
            correct_inverse_frequency_prior_rowwise(
                raw_probability[:, block_slice],
                probe_fit_prior[:, block_slice],
                epsilon=epsilon,
            )
        )
    return np.concatenate(corrected, axis=1)


def _oof_probe_scores(
    cohort: EvidenceCohort,
    rows: np.ndarray,
    config: D1Config,
    *,
    seed_parts: tuple[object, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[dict[str, object], ...]]:
    """Generate probe OOF scores inside an already isolated fit population."""

    rows = np.asarray(rows, dtype=np.int64)
    _, widths = _concept_layout(cohort)
    dimension = sum(widths)
    splits = case_kfold_splits(
        len(rows),
        config.inner_folds,
        _seed(config.seed, *seed_parts, "split"),
    )
    _validate_partition(splits, len(rows))
    raw = np.full((len(rows), dimension), np.nan, dtype=np.float64)
    prior = np.full_like(raw, np.nan)
    validation_fold = np.full(len(rows), -1, dtype=np.int64)
    manifests: list[dict[str, object]] = []
    for inner_fold, (fit_local, validation_local) in enumerate(splits):
        probe_fit = rows[fit_local]
        probe_validation = rows[validation_local]
        fold_raw, fold_prior = _fit_probe_blocks(
            cohort,
            probe_fit,
            probe_validation,
            config,
            (*seed_parts, "probe", inner_fold),
        )
        raw[validation_local] = fold_raw
        prior[validation_local] = fold_prior
        validation_fold[validation_local] = inner_fold
        manifests.append(
            {
                "inner_fold": int(inner_fold),
                "fit_count": int(len(probe_fit)),
                "validation_count": int(len(probe_validation)),
                "fit_case_id_sha256": _case_hash(cohort.case_id[probe_fit]),
                "validation_case_id_sha256": _case_hash(
                    cohort.case_id[probe_validation]
                ),
                "fit_validation_disjoint": True,
                "split_inputs": "case_indices_only_no_C_no_Y",
            }
        )
    if (
        not np.isfinite(raw).all()
        or not np.isfinite(prior).all()
        or np.any(validation_fold < 0)
    ):
        raise AssertionError("inner OOF probe predictions are incomplete")
    return raw, prior, validation_fold, tuple(manifests)


def _fit_temperature_blocks(
    corrected_fit: np.ndarray,
    concept_labels: np.ndarray,
    corrected_predict: np.ndarray,
    names: Sequence[str],
    widths: Sequence[int],
    config: D1Config,
) -> tuple[np.ndarray, tuple[float, ...], tuple[dict[str, object], ...]]:
    """Fit concept-only temperatures and transform a wholly held-out block."""

    transformed = np.full_like(corrected_predict, np.nan)
    temperatures: list[float] = []
    manifests: list[dict[str, object]] = []
    for concept, (name, block_slice) in enumerate(
        zip(names, _block_slices(widths), strict=True)
    ):
        calibrator = ScalarTemperatureCalibrator(
            epsilon=config.probability_epsilon,
            temperature_bounds=(config.temperature_lower, config.temperature_upper),
            optimization_tolerance=config.temperature_optimization_tolerance,
        ).fit(corrected_fit[:, block_slice], concept_labels[:, concept])
        transformed[:, block_slice] = calibrator.transform(
            corrected_predict[:, block_slice]
        )
        temperature = float(calibrator.temperature_)
        temperatures.append(temperature)
        manifests.append(
            {
                "concept": str(name),
                "temperature": temperature,
                "nll_at_unit_temperature": float(
                    calibrator.nll_at_unit_temperature_
                ),
                "training_nll": float(calibrator.training_nll_),
                "optimization_success": bool(calibrator.optimization_success_),
                "endpoint_used": "concept_label_C_only_no_diagnosis_Y",
            }
        )
    if not np.isfinite(transformed).all():
        raise AssertionError("temperature transformation is incomplete")
    return transformed, tuple(temperatures), tuple(manifests)


def build_nested_concept_scores(
    cohort: EvidenceCohort,
    fit_rows: np.ndarray,
    validation_rows: np.ndarray,
    config: D1Config,
    *,
    repetition: int,
    outer_fold: int,
) -> NestedConceptScores:
    """Generate strictly meta-heldout train and outer-validation marginals.

    A temperature model for meta-heldout block ``H`` is fitted only inside
    ``A = outer_train \\ H``.  Its training scores are themselves inner-OOF
    within ``A``.  The probe that predicts ``H`` is also fitted on ``A`` and
    the prior correction uses ``A``.  Consequently every final row in ``H`` is
    invariant to arbitrary changes of concept or diagnosis labels anywhere in
    ``H``; there is no indirect path through another row's probe score.
    """

    fit_rows = np.asarray(fit_rows, dtype=np.int64)
    validation_rows = np.asarray(validation_rows, dtype=np.int64)
    if np.intersect1d(fit_rows, validation_rows).size:
        raise ValueError("outer fit and validation rows overlap")
    names, widths = _concept_layout(cohort)
    dimension = sum(widths)
    meta_splits = case_kfold_splits(
        len(fit_rows),
        config.calibration_folds,
        _seed(config.seed, "strict_meta", repetition, outer_fold),
    )
    _validate_partition(meta_splits, len(fit_rows))
    train_raw = np.full((len(fit_rows), dimension), np.nan, dtype=np.float64)
    train_prior = np.full_like(train_raw, np.nan)
    train_calibrated = np.full_like(train_raw, np.nan)
    meta_fold_id = np.full(len(fit_rows), -1, dtype=np.int64)
    meta_manifests: list[dict[str, object]] = []
    for meta_fold, (meta_fit_local, meta_heldout_local) in enumerate(meta_splits):
        meta_fit = fit_rows[meta_fit_local]
        meta_heldout = fit_rows[meta_heldout_local]
        meta_oof_raw, meta_oof_prior, _, meta_inner_manifests = _oof_probe_scores(
            cohort,
            meta_fit,
            config,
            seed_parts=(
                "strict_meta_inner",
                repetition,
                outer_fold,
                meta_fold,
            ),
        )
        meta_oof_corrected = _prior_correct_blocks(
            meta_oof_raw,
            meta_oof_prior,
            widths,
            epsilon=config.probability_epsilon,
        )
        heldout_raw, heldout_prior = _fit_probe_blocks(
            cohort,
            meta_fit,
            meta_heldout,
            config,
            ("strict_meta_predict", repetition, outer_fold, meta_fold),
        )
        heldout_corrected = _prior_correct_blocks(
            heldout_raw,
            heldout_prior,
            widths,
            epsilon=config.probability_epsilon,
        )
        heldout_calibrated, temperatures, temperature_manifests = (
            _fit_temperature_blocks(
                meta_oof_corrected,
                _concept_matrix(cohort, meta_fit),
                heldout_corrected,
                names,
                widths,
                config,
            )
        )
        train_raw[meta_heldout_local] = heldout_raw
        train_prior[meta_heldout_local] = heldout_prior
        train_calibrated[meta_heldout_local] = heldout_calibrated
        meta_fold_id[meta_heldout_local] = meta_fold
        meta_manifests.append(
            {
                "meta_fold": int(meta_fold),
                "fit_count": int(len(meta_fit)),
                "heldout_count": int(len(meta_heldout)),
                "fit_case_id_sha256": _case_hash(cohort.case_id[meta_fit]),
                "heldout_case_id_sha256": _case_hash(
                    cohort.case_id[meta_heldout]
                ),
                "fit_heldout_disjoint": True,
                "strict_isolation": (
                    "all probes, probe-fit priors, inner-OOF temperature inputs, "
                    "and temperature labels exclude the entire meta-heldout block"
                ),
                "split_inputs": "case_indices_only_no_C_no_Y",
                "inner_probe_folds_within_meta_fit": list(meta_inner_manifests),
                "temperatures": list(temperatures),
                "concept_temperature_fits": list(temperature_manifests),
            }
        )
    if (
        not np.isfinite(train_raw).all()
        or not np.isfinite(train_prior).all()
        or not np.isfinite(train_calibrated).all()
        or np.any(meta_fold_id < 0)
    ):
        raise AssertionError("strict meta-heldout training predictions are incomplete")

    # Fit the outer-validation temperature on the same top-level strictly
    # meta-heldout score table that downstream heads receive.  Every row in
    # this table was generated by a probe, prior, and temperature-input stack
    # excluding its entire H block.  Reusing it avoids a second OOF estimator
    # with a different information path.  The full outer-train probe below is
    # still the only probe that predicts outer validation.
    outer_oof_corrected = _prior_correct_blocks(
        train_raw,
        train_prior,
        widths,
        epsilon=config.probability_epsilon,
    )
    validation_raw, validation_prior = _fit_probe_blocks(
        cohort,
        fit_rows,
        validation_rows,
        config,
        ("outer_validation_probe_fit_all", repetition, outer_fold),
    )
    validation_corrected = _prior_correct_blocks(
        validation_raw,
        validation_prior,
        widths,
        epsilon=config.probability_epsilon,
    )
    validation_calibrated, validation_temperatures, validation_temperature_fits = (
        _fit_temperature_blocks(
            outer_oof_corrected,
            _concept_matrix(cohort, fit_rows),
            validation_corrected,
            names,
            widths,
            config,
        )
    )
    outer_validation_manifest = (
        {
            "fit_all_for_outer_validation": True,
            "fit_count": int(len(fit_rows)),
            "fit_case_id_sha256": _case_hash(cohort.case_id[fit_rows]),
            "outer_validation_count": int(len(validation_rows)),
            "outer_validation_case_id_sha256": _case_hash(
                cohort.case_id[validation_rows]
            ),
            "endpoint_used": "concept_labels_C_only_no_diagnosis_Y",
            "temperature_fit_score_source": (
                "top_level_strict_metaheldout_prior_corrected_probe_scores"
            ),
            "top_level_metaheldout_fold_count": int(len(meta_manifests)),
            "top_level_metaheldout_case_census": int(len(fit_rows)),
            "temperatures": list(validation_temperatures),
            "concept_temperature_fits": list(validation_temperature_fits),
        }
    )
    return NestedConceptScores(
        train_probability=train_calibrated,
        validation_probability=validation_calibrated,
        train_raw_probability=train_raw,
        validation_raw_probability=validation_raw,
        train_probe_fit_prior=train_prior,
        validation_probe_fit_prior=validation_prior,
        train_meta_fold=meta_fold_id,
        # Compatibility aliases now both identify the wholly held-out meta fold,
        # not two independently generated row folds.
        train_probe_fold=meta_fold_id.copy(),
        train_calibration_fold=meta_fold_id.copy(),
        validation_temperature=tuple(validation_temperatures),
        probability_epsilon=float(config.probability_epsilon),
        inner_manifests=tuple(meta_manifests),
        calibration_manifests=tuple([*meta_manifests, outer_validation_manifest]),
    )


def _per_case_marginal_error(
    joint: np.ndarray,
    marginals: Sequence[np.ndarray],
    states: np.ndarray,
) -> np.ndarray:
    errors = np.zeros(len(joint), dtype=np.float64)
    for concept, target in enumerate(marginals):
        reconstructed = np.column_stack(
            [
                joint[:, states[:, concept] == state].sum(axis=1)
                for state in range(target.shape[1])
            ]
        )
        errors = np.maximum(errors, np.max(np.abs(reconstructed - target), axis=1))
    return errors


def _prediction_rows(
    cohort: EvidenceCohort,
    validation_rows: np.ndarray,
    probabilities: Mapping[str, np.ndarray],
    repetition: int,
    outer_fold: int,
) -> list[dict[str, object]]:
    if tuple(probabilities) != MODEL_NAMES:
        raise AssertionError("D1 model order or census changed")
    output: list[dict[str, object]] = []
    for model in MODEL_NAMES:
        probability = _normalize_task_probability(probabilities[model])
        if probability.shape != (len(validation_rows), len(cohort.labels)):
            raise ValueError(f"invalid diagnosis probability shape for {model}")
        for local, row in enumerate(validation_rows):
            record: dict[str, object] = {
                "dataset_id": cohort.dataset_id,
                "case_id": str(cohort.case_id[row]),
                "metadata_index": int(cohort.metadata_index[row]),
                "label": int(cohort.y[row]),
                "repetition": int(repetition),
                "outer_fold": int(outer_fold),
                "model": model,
            }
            record.update(
                {
                    f"p{class_index}": float(value)
                    for class_index, value in enumerate(probability[local])
                }
            )
            output.append(record)
    return output


def _concept_rows(
    cohort: EvidenceCohort,
    validation_rows: np.ndarray,
    scores: NestedConceptScores,
    repetition: int,
    outer_fold: int,
) -> list[dict[str, object]]:
    names, widths = _concept_layout(cohort)
    output: list[dict[str, object]] = []
    validation_corrected = _prior_correct_blocks(
        scores.validation_raw_probability,
        scores.validation_probe_fit_prior,
        widths,
        epsilon=scores.probability_epsilon,
    )
    for concept, (name, block_slice) in enumerate(
        zip(names, _block_slices(widths), strict=True)
    ):
        raw_probability = scores.validation_raw_probability[:, block_slice]
        prior = scores.validation_probe_fit_prior[:, block_slice]
        corrected_probability = validation_corrected[:, block_slice]
        calibrated_probability = scores.validation_probability[:, block_slice]
        for local, row in enumerate(validation_rows):
            for state in range(widths[concept]):
                output.append(
                    {
                        "dataset_id": cohort.dataset_id,
                        "case_id": str(cohort.case_id[row]),
                        "metadata_index": int(cohort.metadata_index[row]),
                        "diagnosis_label": int(cohort.y[row]),
                        "concept": name,
                        "concept_label": int(cohort.concepts[name][row]),
                        "state": int(state),
                        "repetition": int(repetition),
                        "outer_fold": int(outer_fold),
                        "raw_weighted_probability": float(
                            raw_probability[local, state]
                        ),
                        "probe_fit_prior": float(prior[local, state]),
                        "prior_corrected_probability": float(
                            corrected_probability[local, state]
                        ),
                        "calibrated_probability": float(
                            calibrated_probability[local, state]
                        ),
                        "fit_all_temperature": float(
                            scores.validation_temperature[concept]
                        ),
                    }
                )
    return output


def run_mpjc_pilot(
    cohort: EvidenceCohort,
    config: D1Config = D1Config(),
    *,
    official_test_audit: Mapping[str, object] | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Run D1 on an already quarantined cohort; no external split is accepted."""

    cohort.validate()
    config.validate(len(cohort.y))
    if len(cohort.labels) != 2:
        raise ValueError("the frozen bilinear D1 control currently requires binary Y")
    if official_test_audit is not None:
        if int(official_test_audit.get("official_test_overlap_count", -1)) != 0:
            raise ValueError("official-test audit did not prove zero overlap")
        if bool(official_test_audit.get("official_test_outcomes_opened", True)):
            raise ValueError("official-test outcomes were opened")

    names, widths = _concept_layout(cohort)
    states = enumerate_cartesian_states(widths)
    state_features = _onehot_concepts(states, widths)
    predictions: list[dict[str, object]] = []
    joint_records: list[dict[str, object]] = []
    concept_records: list[dict[str, object]] = []
    fit_manifests: list[dict[str, object]] = []
    n_cases = len(cohort.y)

    for repetition in range(config.repetitions):
        outer_splits = case_kfold_splits(
            n_cases,
            config.outer_folds,
            _seed(config.seed, "outer", repetition),
        )
        _validate_partition(outer_splits, n_cases)
        for outer_fold, (fit_rows, validation_rows) in enumerate(outer_splits):
            nested = build_nested_concept_scores(
                cohort,
                fit_rows,
                validation_rows,
                config,
                repetition=repetition,
                outer_fold=outer_fold,
            )
            train_marginals = _marginal_blocks(nested.train_probability, widths)
            validation_marginals = _marginal_blocks(
                nested.validation_probability, widths
            )
            train_concepts = _concept_matrix(cohort, fit_rows)
            validation_concepts = _concept_matrix(cohort, validation_rows)
            y_fit = np.asarray(cohort.y[fit_rows], dtype=np.int64)

            joint_fit = fit_rank1_joint(
                train_marginals,
                train_concepts,
                l2=config.joint_l2,
                max_iterations=config.joint_max_iterations,
                ipf_tolerance=config.ipf_tolerance,
                ipf_max_iterations=config.ipf_max_iterations,
                n_starts=config.joint_n_starts,
                start_scale=config.joint_start_scale,
                parameter_bound=config.joint_parameter_bound,
            )
            joint_probability, joint_diagnostics = marginal_preserving_joint(
                validation_marginals,
                joint_fit.phi,
                states=states,
                tolerance=config.ipf_tolerance,
                max_iterations=config.ipf_max_iterations,
                return_diagnostics=True,
            )
            independent_probability = independent_product(
                validation_marginals, states=states
            )
            unprojected_fit = fit_unprojected_rank1_joint(
                train_marginals,
                train_concepts,
                l2=config.joint_l2,
                max_iterations=config.joint_max_iterations,
                n_starts=config.joint_n_starts,
                start_scale=config.joint_start_scale,
                parameter_bound=config.joint_parameter_bound,
            )
            unprojected_probability = unprojected_fit.predict(validation_marginals)

            gold_train = _onehot_concepts(train_concepts, widths)
            gold_validation = _onehot_concepts(validation_concepts, widths)
            task_head = _BalancedClassifier(
                len(cohort.labels),
                config.diagnosis_c,
                config.max_iter,
                _seed(config.seed, "task_gold", repetition, outer_fold),
            ).fit(gold_train, y_fit)
            task_table = task_head.predict_proba(state_features)
            validation_hard = _hard_concepts(nested.validation_probability, widths)
            train_hard = _hard_concepts(nested.train_probability, widths)

            soft_head = _BalancedClassifier(
                len(cohort.labels),
                config.diagnosis_c,
                config.max_iter,
                _seed(config.seed, "task_oof_soft", repetition, outer_fold),
            ).fit(nested.train_probability, y_fit)
            hard_head = _BalancedClassifier(
                len(cohort.labels),
                config.diagnosis_c,
                config.max_iter,
                _seed(config.seed, "task_oof_hard", repetition, outer_fold),
            ).fit(train_hard, y_fit)
            direct_head = _BalancedClassifier(
                len(cohort.labels),
                config.diagnosis_c,
                config.max_iter,
                _seed(config.seed, "task_direct_x", repetition, outer_fold),
            ).fit(cohort.x[fit_rows], y_fit)
            bilinear = RankOneBilinearClassifier(
                widths,
                l2=config.bilinear_l2,
                starts=config.bilinear_starts,
                seed=config.bilinear_seed,
                max_iter=config.bilinear_max_iter,
            ).fit(nested.train_probability, y_fit)

            probabilities = {
                "CF_MPJC": exact_task_marginalization(joint_probability, task_table),
                "INDEPENDENT_MARGINALIZATION": exact_task_marginalization(
                    independent_probability, task_table
                ),
                "UNPROJECTED_RANK1_JOINT": exact_task_marginalization(
                    unprojected_probability, task_table
                ),
                "JOINT_MAP": task_table[np.argmax(joint_probability, axis=1)],
                "MARGINAL_MEAN_PLUGIN": task_head.predict_proba(
                    nested.validation_probability
                ),
                "OOF_SOFT_LOGISTIC": soft_head.predict_proba(
                    nested.validation_probability
                ),
                "OOF_HARD_LOGISTIC": hard_head.predict_proba(validation_hard),
                "BILINEAR_SCORE_HEAD": bilinear.predict_proba(
                    nested.validation_probability
                ),
                "GOLD_TO_HARD_PRED": task_head.predict_proba(validation_hard),
                "GOLD_ORACLE": task_head.predict_proba(gold_validation),
                "DIRECT_X": direct_head.predict_proba(cohort.x[validation_rows]),
            }
            predictions.extend(
                _prediction_rows(
                    cohort,
                    validation_rows,
                    probabilities,
                    repetition,
                    outer_fold,
                )
            )
            concept_records.extend(
                _concept_rows(
                    cohort,
                    validation_rows,
                    nested,
                    repetition,
                    outer_fold,
                )
            )

            flat = np.ravel_multi_index(
                validation_concepts.T, widths, mode="raise", order="C"
            )
            joint_margin_error = _per_case_marginal_error(
                joint_probability, validation_marginals, states
            )
            unprojected_margin_drift = _per_case_marginal_error(
                unprojected_probability, validation_marginals, states
            )
            for local, row in enumerate(validation_rows):
                joint_records.append(
                    {
                        "dataset_id": cohort.dataset_id,
                        "case_id": str(cohort.case_id[row]),
                        "metadata_index": int(cohort.metadata_index[row]),
                        "label": int(cohort.y[row]),
                        "repetition": int(repetition),
                        "outer_fold": int(outer_fold),
                        "logp_joint": float(
                            np.log(
                                np.clip(
                                    joint_probability[local, flat[local]], 1e-12, 1.0
                                )
                            )
                        ),
                        "logp_independent": float(
                            np.log(
                                np.clip(
                                    independent_probability[local, flat[local]],
                                    1e-12,
                                    1.0,
                                )
                            )
                        ),
                        "logp_unprojected": float(
                            np.log(
                                np.clip(
                                    unprojected_probability[local, flat[local]],
                                    1e-12,
                                    1.0,
                                )
                            )
                        ),
                        "max_joint_marginal_error": float(joint_margin_error[local]),
                        "max_unprojected_marginal_error": float(
                            unprojected_margin_drift[local]
                        ),
                    }
                )

            if bilinear.summary_ is None:
                raise AssertionError("bilinear baseline lacks a fit summary")
            fit_manifests.append(
                {
                    "repetition": int(repetition),
                    "outer_fold": int(outer_fold),
                    "fit_count": int(len(fit_rows)),
                    "validation_count": int(len(validation_rows)),
                    "fit_case_id_sha256": _case_hash(cohort.case_id[fit_rows]),
                    "validation_case_id_sha256": _case_hash(
                        cohort.case_id[validation_rows]
                    ),
                    "outer_split_inputs": "case_indices_only_no_C_no_Y",
                    "joint_fit_endpoint": "joint_concept_C_only_no_diagnosis_Y",
                    "joint_converged": bool(joint_fit.converged),
                    "joint_objective": float(joint_fit.objective),
                    "joint_concept_nll": float(joint_fit.concept_nll),
                    "joint_iterations": int(joint_fit.iterations),
                    "joint_gradient_norm": float(joint_fit.gradient_norm),
                    "joint_phi": joint_fit.phi.astype(float).tolist(),
                    "joint_projection_converged": bool(joint_diagnostics.converged),
                    "joint_projection_iterations": int(joint_diagnostics.iterations),
                    "joint_projection_solver": str(joint_diagnostics.solver),
                    "joint_projection_stages": int(joint_diagnostics.stages),
                    "joint_projection_stage_iterations": [
                        int(value) for value in joint_diagnostics.stage_iterations
                    ],
                    "joint_projection_energy_span": float(
                        joint_diagnostics.energy_span
                    ),
                    "joint_projection_max_marginal_error": float(
                        joint_diagnostics.max_marginal_error
                    ),
                    "max_joint_marginal_error": float(np.max(joint_margin_error)),
                    "unprojected_fit_endpoint": "joint_concept_C_only_no_diagnosis_Y",
                    "unprojected_converged": bool(unprojected_fit.converged),
                    "unprojected_objective": float(unprojected_fit.objective),
                    "unprojected_concept_nll": float(unprojected_fit.concept_nll),
                    "unprojected_iterations": int(unprojected_fit.iterations),
                    "unprojected_gradient_norm": float(unprojected_fit.gradient_norm),
                    "unprojected_phi": unprojected_fit.phi.astype(float).tolist(),
                    "max_unprojected_marginal_drift": float(
                        np.max(unprojected_margin_drift)
                    ),
                    "shared_joint_hyperparameters": {
                        "l2": float(config.joint_l2),
                        "max_iterations": int(config.joint_max_iterations),
                        "n_starts": int(config.joint_n_starts),
                        "start_scale": float(config.joint_start_scale),
                        "parameter_bound": float(config.joint_parameter_bound),
                    },
                    "bilinear_converged": bool(bilinear.summary_.converged),
                    "bilinear_objective": float(bilinear.summary_.objective),
                    "bilinear_iterations": int(bilinear.summary_.iterations),
                    "bilinear_selected_start": int(bilinear.summary_.start_index),
                    "bilinear_parameter_count": int(bilinear.summary_.parameter_count),
                    "strict_metaheldout_isolation": True,
                    "outer_validation_temperature_top_level_meta_folds": list(
                        nested.inner_manifests
                    ),
                    "strict_meta_calibration_folds": list(
                        nested.calibration_manifests
                    ),
                }
            )
            if progress is not None:
                progress(
                    f"[CF-MPJC D1] repetition {repetition + 1}/{config.repetitions}, "
                    f"outer fold {outer_fold + 1}/{config.outer_folds} complete"
                )

    expected_prediction_rows = n_cases * config.repetitions * len(MODEL_NAMES)
    if len(predictions) != expected_prediction_rows:
        raise AssertionError("D1 diagnosis prediction census is incomplete")
    if len(joint_records) != n_cases * config.repetitions:
        raise AssertionError("D1 joint diagnostic census is incomplete")
    if len(concept_records) != n_cases * config.repetitions * sum(widths):
        raise AssertionError("D1 concept prediction census is incomplete")
    return {
        "schema_version": "1.0.0",
        "protocol_id": "DERM7PT_CF_MPJC_D1_V1_20260908",
        "evidence_status": "CONTAMINATED_CASE_LEVEL_DEVELOPMENT_METHOD_SCREEN",
        "dataset_id": cohort.dataset_id,
        "labels": list(cohort.labels),
        "concept_states": {name: list(cohort.concept_states[name]) for name in names},
        "models": list(MODEL_NAMES),
        "config": asdict(config),
        "case_independence_unit": "case_no_patient_identifier_released",
        "outer_split_inputs": "case_indices_only_no_C_no_Y",
        "joint_fit_uses_diagnosis_endpoint": False,
        "official_test_used": False,
        "official_test_audit": dict(official_test_audit or {}),
        "fit_manifests": fit_manifests,
        "predictions": predictions,
        "joint_records": joint_records,
        "concept_records": concept_records,
    }


__all__ = [
    "D1Config",
    "MODEL_NAMES",
    "NestedConceptScores",
    "build_nested_concept_scores",
    "case_kfold_splits",
    "run_mpjc_pilot",
    "validate_official_test_exclusion",
]
