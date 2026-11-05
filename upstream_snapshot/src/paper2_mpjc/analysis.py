"""Case-clustered analysis and frozen success gates for Derm7pt CF-MPJC D1.

Repeated cross-validation produces several out-of-fold predictions for the same
case.  Those predictions are repeated measurements, not independent cases and
not an automatically deployable ensemble.  All inferential routines below
therefore pair models by ``(case_id, repetition)``, average the estimand across
repetitions, and bootstrap cases (stratified by diagnosis when the estimand is
diagnostic).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score


PRIMARY_MODELS = (
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
class MPJCBootstrapConfig:
    draws: int = 10000
    seed: int = 2026090831
    noninferiority_nats: float = 0.005
    tail_fraction: float = 0.10
    concentration_top_fraction: float = 0.10
    concentration_max_share: float = 0.80


def _validate_config(config: MPJCBootstrapConfig) -> None:
    if config.draws < 1:
        raise ValueError("draws must be positive")
    if config.noninferiority_nats < 0.0:
        raise ValueError("noninferiority_nats must be non-negative")
    if not 0.0 < config.tail_fraction <= 1.0:
        raise ValueError("tail_fraction must lie in (0, 1]")
    if not 0.0 < config.concentration_top_fraction < 1.0:
        raise ValueError("concentration_top_fraction must lie in (0, 1)")
    if not 0.0 < config.concentration_max_share <= 1.0:
        raise ValueError("concentration_max_share must lie in (0, 1]")


def _probability_columns(table: pd.DataFrame) -> list[str]:
    columns = sorted(
        (
            column
            for column in table.columns
            if isinstance(column, str)
            and column.startswith("p")
            and column[1:].isdigit()
        ),
        key=lambda value: int(value[1:]),
    )
    expected = [f"p{index}" for index in range(len(columns))]
    if len(columns) < 2 or columns != expected:
        raise ValueError("probability columns must be contiguous p0,...,pK-1")
    return columns


def _validate_probability(probability: np.ndarray) -> np.ndarray:
    probability = np.asarray(probability, dtype=np.float64)
    if probability.ndim != 2 or probability.shape[0] == 0:
        raise ValueError("probability matrix is invalid")
    if not np.isfinite(probability).all():
        raise ValueError("probability matrix contains non-finite values")
    if np.any(probability < -1e-12):
        raise ValueError("probability matrix contains negative mass")
    row_sum = probability.sum(axis=1)
    if not np.allclose(row_sum, 1.0, rtol=0.0, atol=2e-6):
        raise ValueError("probability rows do not lie on the simplex")
    # Correct only floating-point dust admitted by the checks above.
    probability = np.maximum(probability, 0.0)
    return probability / probability.sum(axis=1, keepdims=True)


def _validate_labels(y: np.ndarray, class_count: int) -> np.ndarray:
    raw = np.asarray(y)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("labels must be a non-empty vector")
    try:
        numeric = raw.astype(np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("labels must be integer class indices") from error
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError("labels must be integer class indices")
    labels = numeric.astype(np.int64)
    if np.any(labels < 0) or np.any(labels >= class_count):
        raise ValueError("labels lie outside the probability columns")
    return labels


def _validate_prediction_rows(predictions: pd.DataFrame) -> list[str]:
    required = {"case_id", "label", "repetition", "model"}
    if not required.issubset(predictions.columns):
        raise ValueError("prediction table is missing identifier columns")
    if predictions.empty:
        raise ValueError("prediction table is empty")
    if predictions[["case_id", "label", "repetition", "model"]].isna().any().any():
        raise ValueError("prediction identifiers contain missing values")
    probability_columns = _probability_columns(predictions)
    _validate_probability(predictions[probability_columns].to_numpy(dtype=np.float64))
    _validate_labels(predictions["label"].to_numpy(), len(probability_columns))
    if predictions.groupby("case_id", sort=False)["label"].nunique().max() != 1:
        raise ValueError("a case has inconsistent diagnosis labels")
    if predictions.duplicated(["case_id", "repetition", "model"]).any():
        raise ValueError("prediction rows are duplicated")
    if "outer_fold" in predictions:
        if predictions["outer_fold"].isna().any():
            raise ValueError("outer-fold identifiers contain missing values")
        fold_count = predictions.groupby(
            ["case_id", "repetition"], sort=False
        )["outer_fold"].nunique()
        if int(fold_count.max()) != 1:
            raise ValueError("models assign a case/repetition pair to different outer folds")
    case_values = pd.unique(predictions["case_id"])
    if len({str(value) for value in case_values}) != len(case_values):
        raise ValueError("case identifiers collide after serialization")
    return probability_columns


def macro_true_log_probability(y: np.ndarray, probability: np.ndarray) -> float:
    probability = _validate_probability(probability)
    y = _validate_labels(y, probability.shape[1])
    if len(y) != len(probability):
        raise ValueError("labels and probabilities have different lengths")
    logp = np.log(np.clip(probability[np.arange(len(y)), y], 1e-12, 1.0))
    return float(np.mean([logp[y == label].mean() for label in np.unique(y)]))


def _top_label_ece(
    y: np.ndarray,
    probability: np.ndarray,
    *,
    bins: int,
    sample_weight: np.ndarray,
) -> float:
    probability = _validate_probability(probability)
    y = _validate_labels(y, probability.shape[1])
    if len(y) != len(probability):
        raise ValueError("labels and probabilities have different lengths")
    if bins < 1:
        raise ValueError("bins must be positive")
    sample_weight = np.asarray(sample_weight, dtype=np.float64)
    if sample_weight.shape != (len(y),) or not np.isfinite(sample_weight).all():
        raise ValueError("sample weights are invalid")
    if np.any(sample_weight < 0.0) or sample_weight.sum() <= 0.0:
        raise ValueError("sample weights must have positive mass")
    sample_weight = sample_weight / sample_weight.sum()
    confidence = probability.max(axis=1)
    correct = (probability.argmax(axis=1) == y).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    output = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            mask = (confidence >= edges[index]) & (confidence < edges[index + 1])
        bin_mass = float(sample_weight[mask].sum())
        if bin_mass > 0.0:
            accuracy = float(np.dot(sample_weight[mask], correct[mask]) / bin_mass)
            mean_confidence = float(
                np.dot(sample_weight[mask], confidence[mask]) / bin_mass
            )
            output += bin_mass * abs(accuracy - mean_confidence)
    return float(output)


def expected_calibration_error(
    y: np.ndarray, probability: np.ndarray, bins: int = 10
) -> float:
    """Empirical-prevalence top-label ECE (descriptive only in D1)."""

    y_array = np.asarray(y)
    return _top_label_ece(
        y_array,
        probability,
        bins=bins,
        sample_weight=np.ones(len(y_array), dtype=np.float64),
    )


def class_balanced_expected_calibration_error(
    y: np.ndarray, probability: np.ndarray, bins: int = 10
) -> float:
    """Top-label ECE under the equal-diagnosis-prior D1 target distribution."""

    probability = _validate_probability(probability)
    labels = _validate_labels(y, probability.shape[1])
    classes, counts = np.unique(labels, return_counts=True)
    count_lookup = dict(zip(classes.tolist(), counts.tolist(), strict=True))
    weights = np.asarray(
        [1.0 / (len(classes) * count_lookup[int(label)]) for label in labels],
        dtype=np.float64,
    )
    return _top_label_ece(labels, probability, bins=bins, sample_weight=weights)


def _upper_tail_mean(values: np.ndarray, fraction: float) -> float:
    """Empirical upper-tail mean with a fractional boundary observation."""

    values = np.sort(np.asarray(values, dtype=np.float64))[::-1]
    target_mass = fraction * len(values)
    whole = int(np.floor(target_mass))
    remainder = target_mass - whole
    total = float(values[:whole].sum())
    if remainder > 0.0:
        total += remainder * float(values[whole])
    return total / target_mass


def class_balanced_cvar_nll(
    y: np.ndarray, probability: np.ndarray, tail_fraction: float = 0.10
) -> float:
    """Macro-average the exact empirical worst-tail NLL within each class."""

    if not 0.0 < tail_fraction <= 1.0:
        raise ValueError("tail_fraction must lie in (0, 1]")
    probability = _validate_probability(probability)
    y = _validate_labels(y, probability.shape[1])
    if len(y) != len(probability):
        raise ValueError("labels and probabilities have different lengths")
    loss = -np.log(np.clip(probability[np.arange(len(y)), y], 1e-12, 1.0))
    return float(
        np.mean(
            [
                _upper_tail_mean(loss[y == label], tail_fraction)
                for label in np.unique(y)
            ]
        )
    )


def _model_panel(
    predictions: pd.DataFrame, model: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    probability_columns = _validate_prediction_rows(predictions)
    subset = predictions.loc[predictions["model"] == model].copy()
    if subset.empty:
        raise ValueError(f"model {model} is absent")
    repetitions = np.asarray(sorted(pd.unique(subset["repetition"])), dtype=object)
    cases = np.asarray(
        sorted(pd.unique(subset["case_id"]), key=lambda value: str(value)), dtype=object
    )
    expected_pairs = {(case, repetition) for case in cases for repetition in repetitions}
    observed_pairs = set(zip(subset["case_id"], subset["repetition"], strict=True))
    if observed_pairs != expected_pairs:
        raise ValueError(f"model {model} does not have a rectangular case/repetition panel")
    case_location = {case: index for index, case in enumerate(cases)}
    repetition_location = {
        repetition: index for index, repetition in enumerate(repetitions)
    }
    probability = np.empty(
        (len(repetitions), len(cases), len(probability_columns)), dtype=np.float64
    )
    label_by_case = (
        subset[["case_id", "label"]]
        .drop_duplicates("case_id")
        .set_index("case_id")["label"]
    )
    y = np.asarray([int(label_by_case.loc[case]) for case in cases], dtype=np.int64)
    for row in subset.itertuples(index=False):
        case_index = case_location[row.case_id]
        repetition_index = repetition_location[row.repetition]
        row_label = int(row.label)
        if y[case_index] != row_label:
            raise ValueError("a case has inconsistent diagnosis labels")
        probability[repetition_index, case_index] = [
            float(getattr(row, column)) for column in probability_columns
        ]
    probability = _validate_probability(
        probability.reshape(-1, probability.shape[-1])
    ).reshape(probability.shape)
    return cases.astype(str), y, repetitions, probability


def _paired_panel(
    predictions: pd.DataFrame, candidate: str, baseline: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if candidate == baseline:
        raise ValueError("candidate and baseline must differ")
    candidate_case, candidate_y, candidate_rep, candidate_probability = _model_panel(
        predictions.loc[predictions["model"].isin([candidate, baseline])], candidate
    )
    baseline_case, baseline_y, baseline_rep, baseline_probability = _model_panel(
        predictions.loc[predictions["model"].isin([candidate, baseline])], baseline
    )
    if (
        not np.array_equal(candidate_case, baseline_case)
        or not np.array_equal(candidate_y, baseline_y)
        or not np.array_equal(candidate_rep, baseline_rep)
    ):
        raise ValueError("candidate and baseline case/repetition panels are not aligned")
    return (
        candidate_case,
        candidate_y,
        candidate_rep,
        candidate_probability,
        baseline_probability,
    )


def _auc(y: np.ndarray, probability: np.ndarray) -> float:
    try:
        if probability.shape[1] == 2:
            return float(roc_auc_score(y, probability[:, 1]))
        return float(
            roc_auc_score(
                np.eye(probability.shape[1])[y],
                probability,
                average="macro",
                multi_class="ovr",
            )
        )
    except ValueError:
        return float("nan")


def model_metrics(
    predictions: pd.DataFrame, model: str, tail_fraction: float
) -> dict[str, float]:
    """Mean of independently evaluated repetition-level metrics (no ensembling)."""

    _, y, _, probability = _model_panel(predictions, model)
    by_repetition: dict[str, list[float]] = {
        "macro_nll": [],
        "balanced_accuracy": [],
        "macro_f1": [],
        "macro_auc_ovr": [],
        "empirical_ece10": [],
        "class_balanced_ece10": [],
        "class_balanced_cvar_nll": [],
    }
    for repetition_probability in probability:
        predicted = repetition_probability.argmax(axis=1)
        by_repetition["macro_nll"].append(
            -macro_true_log_probability(y, repetition_probability)
        )
        by_repetition["balanced_accuracy"].append(
            float(balanced_accuracy_score(y, predicted))
        )
        by_repetition["macro_f1"].append(
            float(f1_score(y, predicted, average="macro", zero_division=0))
        )
        by_repetition["macro_auc_ovr"].append(_auc(y, repetition_probability))
        by_repetition["empirical_ece10"].append(
            expected_calibration_error(y, repetition_probability, bins=10)
        )
        by_repetition["class_balanced_ece10"].append(
            class_balanced_expected_calibration_error(
                y, repetition_probability, bins=10
            )
        )
        by_repetition["class_balanced_cvar_nll"].append(
            class_balanced_cvar_nll(
                y, repetition_probability, tail_fraction=tail_fraction
            )
        )
    return {
        name: float(np.nanmean(np.asarray(values, dtype=np.float64)))
        for name, values in by_repetition.items()
    }


def _stratified_mean_bootstrap(
    values: np.ndarray,
    strata: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> dict[str, object]:
    values = np.asarray(values, dtype=np.float64)
    strata = np.asarray(strata)
    if values.ndim != 1 or strata.ndim != 1 or len(values) != len(strata):
        raise ValueError("bootstrap values and strata are not aligned")
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("bootstrap values are empty or non-finite")
    if draws < 1:
        raise ValueError("draws must be positive")
    classes = np.unique(strata)
    locations = [np.flatnonzero(strata == label) for label in classes]
    point = float(np.mean([values[index].mean() for index in locations]))
    rng = np.random.default_rng(seed)
    sampled = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        sampled[draw] = np.mean(
            [
                values[rng.choice(index, len(index), replace=True)].mean()
                for index in locations
            ]
        )
    return {
        "point": point,
        "ci90": np.quantile(sampled, [0.05, 0.95]).astype(float).tolist(),
        "ci95": np.quantile(sampled, [0.025, 0.975]).astype(float).tolist(),
        "draws": int(draws),
        "seed": int(seed),
    }


def _case_log_probability_gain(
    predictions: pd.DataFrame, candidate: str, baseline: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    case_id, y, repetitions, candidate_probability, baseline_probability = _paired_panel(
        predictions, candidate, baseline
    )
    row = np.arange(len(y))
    gain = np.empty((len(repetitions), len(y)), dtype=np.float64)
    for repetition_index in range(len(repetitions)):
        gain[repetition_index] = np.log(
            np.clip(candidate_probability[repetition_index, row, y], 1e-12, 1.0)
        ) - np.log(
            np.clip(baseline_probability[repetition_index, row, y], 1e-12, 1.0)
        )
    return case_id, y, repetitions, gain


def paired_log_probability_gain(
    predictions: pd.DataFrame,
    candidate: str,
    baseline: str,
    *,
    draws: int,
    seed: int,
) -> dict[str, object]:
    case_id, y, repetitions, gain = _case_log_probability_gain(
        predictions, candidate, baseline
    )
    del case_id
    per_repetition = [
        float(
            np.mean(
                [
                    gain[index, y == label].mean()
                    for label in np.unique(y)
                ]
            )
        )
        for index in range(len(repetitions))
    ]
    case_gain = gain.mean(axis=0)
    summary = _stratified_mean_bootstrap(
        case_gain,
        y,
        draws=draws,
        seed=seed,
    )
    summary.update(
        {
            "candidate": candidate,
            "baseline": baseline,
            "estimand": (
                "diagnosis-macro true-class log-probability gain, averaged within "
                "case across repeated CV"
            ),
            "per_repetition_gain": per_repetition,
            "positive_repetition_fraction": float(
                np.mean(np.asarray(per_repetition) > 0.0)
            ),
            "inference_unit": "case; diagnosis-stratified case bootstrap",
        }
    )
    return summary


def paired_metric_improvement(
    predictions: pd.DataFrame,
    candidate: str,
    baseline: str,
    metric: Callable[[np.ndarray, np.ndarray], float],
    *,
    metric_name: str,
    draws: int,
    seed: int,
) -> dict[str, object]:
    """Bootstrap mean repetition-level ``baseline - candidate`` metric."""

    _, y, repetitions, candidate_probability, baseline_probability = _paired_panel(
        predictions, candidate, baseline
    )
    if draws < 1:
        raise ValueError("draws must be positive")

    def improvement(index: np.ndarray) -> float:
        return float(
            np.mean(
                [
                    metric(y[index], baseline_probability[rep, index])
                    - metric(y[index], candidate_probability[rep, index])
                    for rep in range(len(repetitions))
                ]
            )
        )

    complete_index = np.arange(len(y))
    point = improvement(complete_index)
    classes = np.unique(y)
    locations = [np.flatnonzero(y == label) for label in classes]
    rng = np.random.default_rng(seed)
    sampled = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        # The same sampled cases (with all their repetitions) are used for both models.
        index = np.concatenate(
            [rng.choice(location, len(location), replace=True) for location in locations]
        )
        sampled[draw] = improvement(index)
    return {
        "candidate": candidate,
        "baseline": baseline,
        "metric": metric_name,
        "estimand": (
            "mean repetition-level baseline minus candidate; positive favors candidate"
        ),
        "point": point,
        "ci90": np.quantile(sampled, [0.05, 0.95]).astype(float).tolist(),
        "ci95": np.quantile(sampled, [0.025, 0.975]).astype(float).tolist(),
        "draws": int(draws),
        "seed": int(seed),
        "inference_unit": "case cluster; diagnosis-stratified case bootstrap",
    }


def _validate_joint_records(joint_records: pd.DataFrame) -> pd.DataFrame:
    required = {
        "case_id",
        "label",
        "repetition",
        "logp_joint",
        "logp_independent",
        "logp_unprojected",
        "max_joint_marginal_error",
        "max_unprojected_marginal_error",
    }
    if not required.issubset(joint_records.columns):
        raise ValueError("joint concept records are incomplete")
    if joint_records.empty or joint_records[list(required)].isna().any().any():
        raise ValueError("joint concept records contain missing values")
    if joint_records.duplicated(["case_id", "repetition"]).any():
        raise ValueError("joint concept records are duplicated")
    numeric_columns = [
        "logp_joint",
        "logp_independent",
        "logp_unprojected",
        "max_joint_marginal_error",
        "max_unprojected_marginal_error",
    ]
    numeric = joint_records[numeric_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("joint concept records contain non-finite numerical values")
    if np.any(
        joint_records[
            ["max_joint_marginal_error", "max_unprojected_marginal_error"]
        ].to_numpy(dtype=np.float64)
        < 0.0
    ):
        raise ValueError("joint concept marginal errors must be non-negative")
    repetitions = pd.unique(joint_records["repetition"])
    cases = pd.unique(joint_records["case_id"])
    expected = {(case, repetition) for case in cases for repetition in repetitions}
    observed = set(
        zip(joint_records["case_id"], joint_records["repetition"], strict=True)
    )
    if observed != expected:
        raise ValueError("joint concept records are not a rectangular repeated-CV panel")
    _validate_labels(joint_records["label"].to_numpy(), 2**31 - 1)
    if joint_records.groupby("case_id", sort=False)["label"].nunique().max() != 1:
        raise ValueError("joint records have inconsistent diagnosis labels")
    return joint_records.copy()


def _joint_logp_comparison(
    table: pd.DataFrame,
    candidate_column: str,
    baseline_column: str,
    *,
    draws: int,
    seed: int,
) -> dict[str, object]:
    working = table.copy()
    working["gain"] = working[candidate_column] - working[baseline_column]
    repetitions = sorted(pd.unique(table["repetition"]))
    per_repetition = [
        float(working.loc[working["repetition"] == repetition, "gain"].mean())
        for repetition in repetitions
    ]
    case = working.groupby("case_id", sort=True)["gain"].mean().reset_index()
    case_gain = case["gain"].to_numpy(dtype=np.float64)
    if draws < 1:
        raise ValueError("draws must be positive")
    rng = np.random.default_rng(seed)
    sampled = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        sampled[draw] = rng.choice(case_gain, len(case_gain), replace=True).mean()
    output: dict[str, object] = {
        "point": float(case_gain.mean()),
        "ci90": np.quantile(sampled, [0.05, 0.95]).astype(float).tolist(),
        "ci95": np.quantile(sampled, [0.025, 0.975]).astype(float).tolist(),
        "per_repetition_gain": per_repetition,
        "positive_repetition_fraction": float(
            np.mean(np.asarray(per_repetition) > 0.0)
        ),
        "estimand": (
            "natural-case held-out gold-concept log-likelihood gain, averaged "
            "within case across repeated CV"
        ),
        "candidate_logp_column": candidate_column,
        "baseline_logp_column": baseline_column,
        "inference_unit": "case; unstratified case bootstrap",
        "draws": int(draws),
        "seed": int(seed),
    }
    if "label" in working:
        label_by_case = (
            working[["case_id", "label"]]
            .drop_duplicates("case_id")
            .set_index("case_id")
            .loc[case["case_id"], "label"]
            .to_numpy(dtype=np.int64)
        )
        macro = _stratified_mean_bootstrap(
            case_gain, label_by_case, draws=draws, seed=seed + 1
        )
        macro["estimand"] = (
            "secondary diagnosis-macro audit of joint concept log-likelihood gain"
        )
        output["diagnosis_macro_audit"] = macro
    return output


def joint_concept_gain(
    joint_records: pd.DataFrame, *, draws: int, seed: int
) -> dict[str, object]:
    table = _validate_joint_records(joint_records)
    projected = _joint_logp_comparison(
        table,
        "logp_joint",
        "logp_independent",
        draws=draws,
        seed=seed,
    )
    projected["comparison_name"] = "marginal-preserving joint vs independent"
    unprojected = _joint_logp_comparison(
        table,
        "logp_unprojected",
        "logp_independent",
        draws=draws,
        seed=seed + 2,
    )
    unprojected["comparison_name"] = "unprojected rank-one joint vs independent"
    projection_cost = _joint_logp_comparison(
        table,
        "logp_joint",
        "logp_unprojected",
        draws=draws,
        seed=seed + 4,
    )
    projection_cost["comparison_name"] = (
        "marginal-preserving joint vs unprojected rank-one joint"
    )
    # Keep the pre-specified projected-vs-independent fields at top level for
    # the success gate; the two attribution comparisons are descriptive only.
    projected["descriptive_attribution"] = {
        "unprojected_vs_independent": unprojected,
        "projected_vs_unprojected": projection_cost,
    }
    return projected


def _fractional_top_sum(values: np.ndarray, fraction: float) -> tuple[float, float]:
    values = np.sort(np.asarray(values, dtype=np.float64))[::-1]
    target_mass = fraction * len(values)
    whole = int(np.floor(target_mass))
    remainder = target_mass - whole
    total = float(values[:whole].sum())
    if remainder > 0.0:
        total += remainder * float(values[whole])
    return total, target_mass


def gain_concentration(
    predictions: pd.DataFrame,
    candidate: str,
    baseline: str,
    *,
    top_fraction: float,
) -> dict[str, object]:
    if not 0.0 < top_fraction < 1.0:
        raise ValueError("top_fraction must lie in (0,1)")
    _, y, _, repeated_gain = _case_log_probability_gain(
        predictions, candidate, baseline
    )
    gain = repeated_gain.mean(axis=0)
    classes = np.unique(y)
    class_gain = {int(label): gain[y == label] for label in classes}
    benefit_by_class = {
        str(label): float(np.mean(values > 0.0))
        for label, values in class_gain.items()
    }
    median_by_class = {
        str(label): float(np.median(values)) for label, values in class_gain.items()
    }
    class_mean = [float(values.mean()) for values in class_gain.values()]
    total = float(np.mean(class_mean))

    # Select the best top_fraction *within each class*.  This is exactly the
    # top fraction of the equal-prior (macro) case mass, unlike a global case
    # count under diagnosis imbalance.
    top_macro_contribution = 0.0
    selected_mass_by_class: dict[str, float] = {}
    for label, values in class_gain.items():
        selected_sum, selected_mass = _fractional_top_sum(values, top_fraction)
        top_macro_contribution += selected_sum / len(values) / len(classes)
        selected_mass_by_class[str(label)] = float(selected_mass)
    denominator_nonpositive = bool(total <= 0.0)
    top_share = (
        None
        if denominator_nonpositive
        else float(top_macro_contribution / total)
    )
    return {
        "gain_definition": (
            "per-case mean repeated-CV true-class log-probability gain"
        ),
        "overall_median_gain": float(np.median(gain)),
        "class_median_gain": median_by_class,
        "class_balanced_median_gain": float(np.mean(list(median_by_class.values()))),
        "minimum_class_median_gain": float(np.min(list(median_by_class.values()))),
        "class_benefit_fraction": benefit_by_class,
        "class_balanced_benefit_fraction": float(
            np.mean(list(benefit_by_class.values()))
        ),
        "minimum_class_benefit_fraction": float(
            np.min(list(benefit_by_class.values()))
        ),
        "class_balanced_net_gain": total,
        "top_fraction": float(top_fraction),
        "top_fraction_selection": "within each diagnosis class with fractional boundary",
        "selected_case_mass_by_class": selected_mass_by_class,
        "top_fraction_macro_net_gain_contribution": float(top_macro_contribution),
        "denominator_nonpositive": denominator_nonpositive,
        "top_fraction_net_gain_share": top_share,
    }


def _noninferiority_pass(comparison: dict[str, object], margin_nats: float) -> bool:
    """One-sided alpha=.05 NI test via the lower endpoint of a two-sided 90% CI."""

    return bool(float(comparison["ci90"][0]) >= -margin_nats)


def analyze_mpjc_d1(
    predictions: pd.DataFrame,
    joint_records: pd.DataFrame,
    fit_manifests: Sequence[dict[str, object]],
    *,
    models: Sequence[str] = PRIMARY_MODELS,
    config: MPJCBootstrapConfig = MPJCBootstrapConfig(),
) -> dict[str, object]:
    _validate_config(config)
    models = tuple(models)
    if len(models) != len(set(models)):
        raise ValueError("model census contains duplicate names")
    probability_columns = _validate_prediction_rows(predictions)
    del probability_columns
    if "outer_fold" not in predictions:
        raise ValueError("prediction table omits outer_fold")
    required_for_analysis = set(PRIMARY_MODELS)
    if not required_for_analysis.issubset(models):
        raise ValueError("models omit a pre-specified D1 comparator")
    repetitions = int(predictions["repetition"].nunique())
    cases = int(predictions["case_id"].nunique())
    expected = repetitions * cases
    census = predictions.groupby("model", sort=False).size().to_dict()
    if set(census) != set(models) or any(
        int(value) != expected for value in census.values()
    ):
        raise ValueError("D1 prediction census is incomplete")
    for model in models:
        _model_panel(predictions, model)

    joint_table = _validate_joint_records(joint_records)
    prediction_pairs = set(
        zip(predictions["case_id"], predictions["repetition"], strict=True)
    )
    joint_pairs = set(
        zip(joint_table["case_id"], joint_table["repetition"], strict=True)
    )
    if joint_pairs != prediction_pairs:
        raise ValueError("joint records and diagnostic predictions cover different cases")
    if "label" in joint_table:
        prediction_label = (
            predictions[["case_id", "label"]]
            .drop_duplicates("case_id")
            .set_index("case_id")["label"]
            .sort_index()
        )
        joint_label = (
            joint_table[["case_id", "label"]]
            .drop_duplicates("case_id")
            .set_index("case_id")["label"]
            .sort_index()
        )
        if not prediction_label.equals(joint_label):
            raise ValueError("joint records and predictions have inconsistent labels")
    observed_labels = np.sort(predictions["label"].astype(np.int64).unique())
    expected_labels = np.arange(len(_probability_columns(predictions)))
    if not np.array_equal(observed_labels, expected_labels):
        raise ValueError("D1 predictions do not contain every diagnosis class")
    marginal_error = joint_table["max_joint_marginal_error"].to_numpy(
        dtype=np.float64
    )
    unprojected_marginal_error = joint_table[
        "max_unprojected_marginal_error"
    ].to_numpy(dtype=np.float64)

    expected_fit_pairs = set(
        predictions[["repetition", "outer_fold"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    if len(fit_manifests) != len(expected_fit_pairs):
        raise ValueError("joint fit manifest census is incomplete")
    convergence_values = []
    manifest_pairs = []
    for item in fit_manifests:
        if "repetition" not in item or "outer_fold" not in item:
            raise ValueError("joint fit manifest omits repetition/outer_fold identity")
        manifest_pairs.append((item["repetition"], item["outer_fold"]))
        value = item.get("joint_converged")
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError("joint_converged must be explicitly boolean")
        convergence_values.append(bool(value))
    if (
        len(set(manifest_pairs)) != len(manifest_pairs)
        or set(manifest_pairs) != expected_fit_pairs
    ):
        raise ValueError("joint fit manifests do not match the outer-fold census")

    comparisons = {
        "mpjc_vs_independent": paired_log_probability_gain(
            predictions,
            "CF_MPJC",
            "INDEPENDENT_MARGINALIZATION",
            draws=config.draws,
            seed=config.seed,
        ),
        "mpjc_vs_joint_map": paired_log_probability_gain(
            predictions,
            "CF_MPJC",
            "JOINT_MAP",
            draws=config.draws,
            seed=config.seed + 1,
        ),
        "mpjc_vs_oof_soft": paired_log_probability_gain(
            predictions,
            "CF_MPJC",
            "OOF_SOFT_LOGISTIC",
            draws=config.draws,
            seed=config.seed + 2,
        ),
        "mpjc_vs_bilinear": paired_log_probability_gain(
            predictions,
            "CF_MPJC",
            "BILINEAR_SCORE_HEAD",
            draws=config.draws,
            seed=config.seed + 3,
        ),
        "mpjc_vs_direct_x": paired_log_probability_gain(
            predictions,
            "CF_MPJC",
            "DIRECT_X",
            draws=config.draws,
            seed=config.seed + 4,
        ),
        "mpjc_vs_unprojected_rank1_joint": paired_log_probability_gain(
            predictions,
            "CF_MPJC",
            "UNPROJECTED_RANK1_JOINT",
            draws=config.draws,
            seed=config.seed + 5,
        ),
    }
    ece_metric = lambda y, p: class_balanced_expected_calibration_error(
        y, p, bins=10
    )
    cvar_metric = lambda y, p: class_balanced_cvar_nll(
        y, p, tail_fraction=config.tail_fraction
    )
    reliability = {
        "class_balanced_ece_vs_oof_soft": paired_metric_improvement(
            predictions,
            "CF_MPJC",
            "OOF_SOFT_LOGISTIC",
            ece_metric,
            metric_name="class-balanced top-label ECE10",
            draws=config.draws,
            seed=config.seed + 10,
        ),
        "class_balanced_ece_vs_bilinear": paired_metric_improvement(
            predictions,
            "CF_MPJC",
            "BILINEAR_SCORE_HEAD",
            ece_metric,
            metric_name="class-balanced top-label ECE10",
            draws=config.draws,
            seed=config.seed + 11,
        ),
        "cvar_vs_oof_soft": paired_metric_improvement(
            predictions,
            "CF_MPJC",
            "OOF_SOFT_LOGISTIC",
            cvar_metric,
            metric_name=f"class-balanced CVaR-{config.tail_fraction:.2f} NLL",
            draws=config.draws,
            seed=config.seed + 12,
        ),
        "cvar_vs_bilinear": paired_metric_improvement(
            predictions,
            "CF_MPJC",
            "BILINEAR_SCORE_HEAD",
            cvar_metric,
            metric_name=f"class-balanced CVaR-{config.tail_fraction:.2f} NLL",
            draws=config.draws,
            seed=config.seed + 13,
        ),
    }
    concept = joint_concept_gain(
        joint_table, draws=config.draws, seed=config.seed + 20
    )
    concentration = gain_concentration(
        predictions,
        "CF_MPJC",
        "INDEPENDENT_MARGINALIZATION",
        top_fraction=config.concentration_top_fraction,
    )

    convergence = float(np.mean(convergence_values))
    maximum_marginal_error = float(np.max(marginal_error))
    maximum_unprojected_marginal_error = float(
        np.max(unprojected_marginal_error)
    )
    concept_pass = bool(concept["ci90"][0] > 0.0)
    diagnostic_pass = bool(comparisons["mpjc_vs_independent"]["ci90"][0] > 0.0)
    marginalization_pass = bool(comparisons["mpjc_vs_joint_map"]["ci90"][0] > 0.0)
    oof_noninferior = _noninferiority_pass(
        comparisons["mpjc_vs_oof_soft"], config.noninferiority_nats
    )
    bilinear_noninferior = _noninferiority_pass(
        comparisons["mpjc_vs_bilinear"], config.noninferiority_nats
    )
    # Pre-specified D1 development gate: either reliability endpoint family may
    # pass, provided it beats both strong controls.  This is exploratory and is
    # not represented as multiplicity-controlled confirmatory inference.
    ece_advantage = bool(
        reliability["class_balanced_ece_vs_oof_soft"]["ci90"][0] > 0.0
        and reliability["class_balanced_ece_vs_bilinear"]["ci90"][0] > 0.0
    )
    cvar_advantage = bool(
        reliability["cvar_vs_oof_soft"]["ci90"][0] > 0.0
        and reliability["cvar_vs_bilinear"]["ci90"][0] > 0.0
    )
    reliability_pass = ece_advantage or cvar_advantage
    distributed_pass = bool(
        concentration["overall_median_gain"] > 0.0
        and concentration["class_balanced_benefit_fraction"] > 0.5
        and not concentration["denominator_nonpositive"]
        and concentration["top_fraction_net_gain_share"] is not None
        and float(concentration["top_fraction_net_gain_share"])
        <= config.concentration_max_share
    )
    numerical_pass = bool(convergence >= 0.95 and maximum_marginal_error <= 1e-8)
    all_pass = all(
        (
            concept_pass,
            diagnostic_pass,
            marginalization_pass,
            oof_noninferior,
            bilinear_noninferior,
            reliability_pass,
            distributed_pass,
            numerical_pass,
        )
    )
    return {
        "schema_version": "1.1.0",
        "analysis_id": "DERM7PT_CF_MPJC_D1_V1_20260908",
        "evidence_status": "CONTAMINATED_CASE_LEVEL_DEVELOPMENT_METHOD_SCREEN",
        "case_count": cases,
        "repetitions": repetitions,
        "model_metrics": {
            model: model_metrics(predictions, model, config.tail_fraction)
            for model in models
        },
        "joint_concept_gain": concept,
        "diagnosis_comparisons": comparisons,
        "comparison_roles": {
            "success_gate": [
                "mpjc_vs_independent",
                "mpjc_vs_joint_map",
                "mpjc_vs_oof_soft",
                "mpjc_vs_bilinear",
            ],
            "descriptive_only": [
                "mpjc_vs_direct_x",
                "mpjc_vs_unprojected_rank1_joint",
            ],
        },
        "reliability_comparisons": reliability,
        "gain_concentration_vs_independent": concentration,
        "numerical_audit": {
            "joint_fit_convergence_fraction": convergence,
            "maximum_joint_marginal_error": maximum_marginal_error,
            "maximum_unprojected_marginal_error": (
                maximum_unprojected_marginal_error
            ),
            "median_unprojected_marginal_error": float(
                np.median(unprojected_marginal_error)
            ),
            "unprojected_marginal_error_role": (
                "descriptive marginal-drift ablation; not a success gate"
            ),
            "joint_fit_manifest_count": len(fit_manifests),
        },
        "gate": {
            "full_joint_concept_nll_ci90_positive": concept_pass,
            "mpjc_vs_independent_diagnosis_ci90_positive": diagnostic_pass,
            "mpjc_vs_joint_map_ci90_positive": marginalization_pass,
            "mpjc_noninferior_to_oof_soft": oof_noninferior,
            "mpjc_noninferior_to_bilinear": bilinear_noninferior,
            "noninferiority_margin_nats": config.noninferiority_nats,
            "noninferiority_rule": (
                "lower endpoint of paired case-bootstrap two-sided 90% CI is "
                "greater than or equal to -margin (one-sided alpha=.05)"
            ),
            "class_balanced_ece_advantage_over_both_strong_heads": ece_advantage,
            "class_balanced_tail_advantage_over_both_strong_heads": cvar_advantage,
            "reliability_union_multiplicity_rule": (
                "Pre-specified exploratory development gate: ECE or CVaR may pass; "
                "each family uses paired 90% LCB > 0 against both controls. This "
                "union is not a multiplicity-controlled confirmatory claim."
            ),
            "at_least_one_reliability_advantage": reliability_pass,
            "benefit_not_top_ten_percent_only": distributed_pass,
            "distribution_rule": (
                "overall median gain > 0, mean within-class benefit fraction > .5, "
                "and top equal-prior 10% net-gain share <= configured maximum"
            ),
            "numerical_projection_and_fit_gate": numerical_pass,
            "derm7pt_initial_method_signal": (
                "PASS" if all_pass else "FAIL_WITH_COMPONENT_DIAGNOSIS_REQUIRED"
            ),
            "second_medical_task_required_before_external_confirmation": True,
        },
        "bootstrap_config": asdict(config),
        "inference_scope": (
            "Diagnosis-stratified case-cluster bootstrap conditional on the fitted "
            "repeated-CV models; repetitions are repeated measurements and are not "
            "treated as independent inferential units."
        ),
        "claim_boundary": (
            "This is a case-level contaminated development screen. It cannot establish "
            "patient-level independence, external validity, clinical utility, or an LLM benefit."
        ),
    }


__all__ = [
    "MPJCBootstrapConfig",
    "PRIMARY_MODELS",
    "analyze_mpjc_d1",
    "class_balanced_cvar_nll",
    "class_balanced_expected_calibration_error",
    "expected_calibration_error",
    "gain_concentration",
    "joint_concept_gain",
    "macro_true_log_probability",
    "model_metrics",
    "paired_log_probability_gain",
    "paired_metric_improvement",
]
