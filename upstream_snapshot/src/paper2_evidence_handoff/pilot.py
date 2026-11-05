"""Nested cross-fitted screen of gold-to-predicted evidence mismatch."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from .data import EvidenceCohort


MODEL_NAMES = (
    "DIRECT_X",
    "GOLD_ORACLE",
    "GOLD_TO_HARD_PRED",
    "GOLD_TO_SOFT_PRED",
    "INSAMPLE_HARD_HANDOFF",
    "OOF_HARD_HANDOFF",
    "INSAMPLE_SOFT_HANDOFF",
    "OOF_SOFT_HANDOFF",
)


@dataclass(frozen=True)
class HandoffConfig:
    seed: int = 2026090817
    outer_folds: int = 5
    inner_folds: int = 4
    repetitions: int = 5
    probe_c: float = 1.0
    diagnosis_c: float = 1.0
    max_iter: int = 2000


class _Classifier:
    def __init__(self, n_classes: int, c: float, max_iter: int, seed: int):
        self.n_classes = int(n_classes)
        self.c = float(c)
        self.max_iter = int(max_iter)
        self.seed = int(seed)
        self.constant: int | None = None
        self.model: LogisticRegression | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "_Classifier":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        observed = np.unique(y)
        if len(observed) == 1:
            self.constant = int(observed[0])
            return self
        counts = np.bincount(y, minlength=self.n_classes).astype(np.float64)
        sample_weight = np.asarray([len(y) / (len(observed) * counts[value]) for value in y])
        self.model = LogisticRegression(
            C=self.c,
            solver="lbfgs",
            multi_class="auto",
            max_iter=self.max_iter,
            random_state=self.seed,
        )
        self.model.fit(x, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        probability = np.zeros((len(x), self.n_classes), dtype=np.float64)
        if self.constant is not None:
            probability[:, self.constant] = 1.0
        elif self.model is not None:
            partial = self.model.predict_proba(x)
            probability[:, self.model.classes_.astype(int)] = partial
        else:
            raise RuntimeError("classifier has not been fit")
        probability = np.clip(probability, 0.0, 1.0)
        probability /= np.clip(probability.sum(axis=1, keepdims=True), 1e-15, None)
        return probability


def _seed(base: int, *parts: object) -> int:
    payload = "|".join([str(base), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(sha256(payload).digest()[:4], "little") & 0x7FFFFFFF


def _hard(probability: np.ndarray, widths: Iterable[int]) -> np.ndarray:
    blocks = []
    start = 0
    for width in widths:
        block = probability[:, start : start + width]
        hard = np.zeros_like(block)
        hard[np.arange(len(block)), np.argmax(block, axis=1)] = 1.0
        blocks.append(hard)
        start += width
    if start != probability.shape[1]:
        raise ValueError("concept widths do not cover the prediction matrix")
    return np.concatenate(blocks, axis=1)


def _fit_probes(
    cohort: EvidenceCohort,
    fit_rows: np.ndarray,
    predict_rows: np.ndarray,
    config: HandoffConfig,
    seed_parts: tuple[object, ...],
) -> np.ndarray:
    blocks = []
    for concept_index, (name, states) in enumerate(cohort.concept_states.items()):
        classifier = _Classifier(
            len(states),
            config.probe_c,
            config.max_iter,
            _seed(config.seed, *seed_parts, concept_index, name),
        ).fit(cohort.x[fit_rows], cohort.concepts[name][fit_rows])
        blocks.append(classifier.predict_proba(cohort.x[predict_rows]))
    return np.concatenate(blocks, axis=1)


def _diagnosis_probability(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_predict: np.ndarray,
    n_classes: int,
    config: HandoffConfig,
    seed_parts: tuple[object, ...],
) -> np.ndarray:
    return _Classifier(
        n_classes,
        config.diagnosis_c,
        config.max_iter,
        _seed(config.seed, *seed_parts),
    ).fit(x_fit, y_fit).predict_proba(x_predict)


def _prediction_rows(
    cohort: EvidenceCohort,
    validation_rows: np.ndarray,
    probabilities: dict[str, np.ndarray],
    repetition: int,
    outer_fold: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model in MODEL_NAMES:
        probability = probabilities[model]
        if probability.shape != (len(validation_rows), len(cohort.labels)):
            raise ValueError(f"invalid probability shape for {model}")
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
                {f"p{class_id}": float(value) for class_id, value in enumerate(probability[local])}
            )
            rows.append(record)
    return rows


def _concept_rows(
    cohort: EvidenceCohort,
    validation_rows: np.ndarray,
    probability: np.ndarray,
    repetition: int,
    outer_fold: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    start = 0
    for name, states in cohort.concept_states.items():
        block = probability[:, start : start + len(states)]
        for local, row in enumerate(validation_rows):
            record: dict[str, object] = {
                "dataset_id": cohort.dataset_id,
                "case_id": str(cohort.case_id[row]),
                "metadata_index": int(cohort.metadata_index[row]),
                "diagnosis_label": int(cohort.y[row]),
                "concept": name,
                "concept_label": int(cohort.concepts[name][row]),
                "repetition": int(repetition),
                "outer_fold": int(outer_fold),
            }
            record.update({f"p{state_id}": float(value) for state_id, value in enumerate(block[local])})
            rows.append(record)
        start += len(states)
    return rows


def run_handoff_pilot(cohort: EvidenceCohort, config: HandoffConfig) -> dict[str, object]:
    """Run repeated nested OOF handoff without touching an external/test split."""

    cohort.validate()
    if config.outer_folds < 2 or config.inner_folds < 2 or config.repetitions < 1:
        raise ValueError("invalid cross-fitting configuration")
    n = len(cohort.y)
    n_classes = len(cohort.labels)
    widths = [len(states) for states in cohort.concept_states.values()]
    gold_all = cohort.gold_features()
    predictions: list[dict[str, object]] = []
    concept_predictions: list[dict[str, object]] = []
    manifests: list[dict[str, object]] = []

    for repetition in range(config.repetitions):
        outer = StratifiedKFold(
            n_splits=config.outer_folds,
            shuffle=True,
            random_state=_seed(config.seed, "outer", repetition),
        )
        for outer_fold, (fit_rows, validation_rows) in enumerate(outer.split(np.zeros(n), cohort.y)):
            fit_rows = np.asarray(fit_rows, dtype=int)
            validation_rows = np.asarray(validation_rows, dtype=int)
            if set(fit_rows) & set(validation_rows):
                raise AssertionError("outer fit/validation overlap")

            inner_probability = np.full((len(fit_rows), cohort.concept_dimension), np.nan)
            inner = StratifiedKFold(
                n_splits=config.inner_folds,
                shuffle=True,
                random_state=_seed(config.seed, "inner", repetition, outer_fold),
            )
            inner_manifest = []
            for inner_fold, (inner_fit_local, inner_validation_local) in enumerate(
                inner.split(np.zeros(len(fit_rows)), cohort.y[fit_rows])
            ):
                inner_fit = fit_rows[np.asarray(inner_fit_local, dtype=int)]
                inner_validation = fit_rows[np.asarray(inner_validation_local, dtype=int)]
                inner_probability[inner_validation_local] = _fit_probes(
                    cohort,
                    inner_fit,
                    inner_validation,
                    config,
                    ("probe", repetition, outer_fold, inner_fold),
                )
                inner_manifest.append(
                    {
                        "inner_fold": inner_fold,
                        "fit_case_id_sha256": sha256("\n".join(sorted(cohort.case_id[inner_fit])).encode()).hexdigest(),
                        "validation_case_id_sha256": sha256("\n".join(sorted(cohort.case_id[inner_validation])).encode()).hexdigest(),
                        "fit_count": int(len(inner_fit)),
                        "validation_count": int(len(inner_validation)),
                    }
                )
            if not np.isfinite(inner_probability).all():
                raise AssertionError("inner OOF concept predictions are incomplete")

            full_fit_probability = _fit_probes(
                cohort,
                fit_rows,
                fit_rows,
                config,
                ("probe_full", repetition, outer_fold),
            )
            validation_probability = _fit_probes(
                cohort,
                fit_rows,
                validation_rows,
                config,
                ("probe_full", repetition, outer_fold),
            )
            # The two preceding calls use identical deterministic fits.  Keeping
            # them separate makes the requested rows explicit and testable.
            inner_hard = _hard(inner_probability, widths)
            full_fit_hard = _hard(full_fit_probability, widths)
            validation_hard = _hard(validation_probability, widths)
            gold_fit = gold_all[fit_rows]
            gold_validation = gold_all[validation_rows]
            y_fit = cohort.y[fit_rows]

            gold_model = _Classifier(
                n_classes,
                config.diagnosis_c,
                config.max_iter,
                _seed(config.seed, "diagnosis", repetition, outer_fold, "gold"),
            ).fit(gold_fit, y_fit)
            probabilities = {
                "DIRECT_X": _diagnosis_probability(
                    cohort.x[fit_rows],
                    y_fit,
                    cohort.x[validation_rows],
                    n_classes,
                    config,
                    ("diagnosis", repetition, outer_fold, "direct"),
                ),
                "GOLD_ORACLE": gold_model.predict_proba(gold_validation),
                "GOLD_TO_HARD_PRED": gold_model.predict_proba(validation_hard),
                "GOLD_TO_SOFT_PRED": gold_model.predict_proba(validation_probability),
                "INSAMPLE_HARD_HANDOFF": _diagnosis_probability(
                    full_fit_hard,
                    y_fit,
                    validation_hard,
                    n_classes,
                    config,
                    ("diagnosis", repetition, outer_fold, "insample_hard"),
                ),
                "OOF_HARD_HANDOFF": _diagnosis_probability(
                    inner_hard,
                    y_fit,
                    validation_hard,
                    n_classes,
                    config,
                    ("diagnosis", repetition, outer_fold, "oof_hard"),
                ),
                "INSAMPLE_SOFT_HANDOFF": _diagnosis_probability(
                    full_fit_probability,
                    y_fit,
                    validation_probability,
                    n_classes,
                    config,
                    ("diagnosis", repetition, outer_fold, "insample_soft"),
                ),
                "OOF_SOFT_HANDOFF": _diagnosis_probability(
                    inner_probability,
                    y_fit,
                    validation_probability,
                    n_classes,
                    config,
                    ("diagnosis", repetition, outer_fold, "oof_soft"),
                ),
            }
            if set(probabilities) != set(MODEL_NAMES):
                raise AssertionError("model census changed")
            predictions.extend(
                _prediction_rows(
                    cohort, validation_rows, probabilities, repetition, outer_fold
                )
            )
            concept_predictions.extend(
                _concept_rows(
                    cohort,
                    validation_rows,
                    validation_probability,
                    repetition,
                    outer_fold,
                )
            )
            manifests.append(
                {
                    "repetition": repetition,
                    "outer_fold": outer_fold,
                    "fit_count": int(len(fit_rows)),
                    "validation_count": int(len(validation_rows)),
                    "fit_case_id_sha256": sha256("\n".join(sorted(cohort.case_id[fit_rows])).encode()).hexdigest(),
                    "validation_case_id_sha256": sha256("\n".join(sorted(cohort.case_id[validation_rows])).encode()).hexdigest(),
                    "inner_folds": inner_manifest,
                }
            )
        print(f"[evidence-handoff] repetition {repetition + 1}/{config.repetitions} complete", flush=True)

    return {
        "schema_version": "1.0.0",
        "protocol_id": "DERM7PT_EVIDENCE_HANDOFF_E0_V1_20260908",
        "evidence_status": "CONTAMINATED_DEVELOPMENT_MECHANISM_SCREEN_ONLY",
        "dataset_id": cohort.dataset_id,
        "labels": list(cohort.labels),
        "concept_states": {name: list(states) for name, states in cohort.concept_states.items()},
        "models": list(MODEL_NAMES),
        "config": asdict(config),
        "official_test_used": False,
        "manifests": manifests,
        "predictions": predictions,
        "concept_predictions": concept_predictions,
    }
