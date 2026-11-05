"""Strict development-only Derm7pt adapter for the evidence-handoff screen.

The official test rows are deliberately excluded before any array is returned.
Dermoscopic BiomedCLIP embeddings are the deployment image ``X``; the seven
expert checklist fields are training-only concepts.  This is a checklist
prototype, not a free-text report or a four-lesson anatomy curriculum.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


LABELS = ("NON_MEL", "MEL")
DIAGNOSIS_GROUPS = {
    "basal cell carcinoma": "BCC",
    "nevus": "NEV",
    "blue nevus": "NEV",
    "clark nevus": "NEV",
    "combined nevus": "NEV",
    "congenital nevus": "NEV",
    "dermal nevus": "NEV",
    "recurrent nevus": "NEV",
    "reed or spitz nevus": "NEV",
    "melanoma": "MEL",
    "melanoma (in situ)": "MEL",
    "melanoma (less than 0.76 mm)": "MEL",
    "melanoma (0.76 to 1.5 mm)": "MEL",
    "melanoma (more than 1.5 mm)": "MEL",
    "melanoma metastasis": "MEL",
    "df/lt/mls/misc": "MISC",
    "dermatofibroma": "MISC",
    "lentigo": "MISC",
    "melanosis": "MISC",
    "miscellaneous": "MISC",
    "vascular lesion": "MISC",
    "seborrheic keratosis": "SK",
}

# Frozen from the public checklist vocabulary.  States absent from the
# development partition are not imported from the already-consumed test split.
CONCEPT_STATES: Mapping[str, tuple[str, ...]] = {
    "pigment_network": ("absent", "typical", "atypical"),
    "blue_whitish_veil": ("absent", "present"),
    "vascular_structures": ("absent", "regular", "irregular"),
    "pigmentation": ("absent", "regular", "irregular"),
    "streaks": ("absent", "regular", "irregular"),
    "dots_and_globules": ("absent", "regular", "irregular"),
    "regression_structures": ("absent", "present"),
}

CONCEPT_VALUE_MAP: Mapping[str, Mapping[str, str]] = {
    "pigment_network": {
        "absent": "absent",
        "typical": "typical",
        "atypical": "atypical",
    },
    "blue_whitish_veil": {"absent": "absent", "present": "present"},
    "vascular_structures": {
        "absent": "absent",
        "arborizing": "regular",
        "comma": "regular",
        "hairpin": "regular",
        "within regression": "regular",
        "wreath": "regular",
        "dotted": "irregular",
        "linear irregular": "irregular",
    },
    "pigmentation": {
        "absent": "absent",
        "diffuse regular": "regular",
        "localized regular": "regular",
        "diffuse irregular": "irregular",
        "localized irregular": "irregular",
    },
    "streaks": {"absent": "absent", "regular": "regular", "irregular": "irregular"},
    "dots_and_globules": {
        "absent": "absent",
        "regular": "regular",
        "irregular": "irregular",
    },
    "regression_structures": {
        "absent": "absent",
        "white areas": "present",
        "blue areas": "present",
        "combinations": "present",
    },
}

EXPECTED_EXCLUSIONS = (564, 799, 823, 826, 832, 833, 838, 843, 850)
EXPECTED_DEVELOPMENT = 609
EXPECTED_DEVELOPMENT_CLASS_COUNTS = (460, 149)
EXPECTED_TEST_AFTER_EXCLUSION = 393
DEVELOPMENT_ONLY_ARCHIVE_ROLE = "DERM7PT_OFFICIAL_TRAIN_VALID_DEV609_ONLY"


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class EvidenceCohort:
    """A case-level image, diagnosis, and training-only concept cohort."""

    dataset_id: str
    case_id: np.ndarray
    metadata_index: np.ndarray
    x: np.ndarray
    y: np.ndarray
    labels: tuple[str, ...]
    concepts: Mapping[str, np.ndarray]
    concept_states: Mapping[str, tuple[str, ...]]

    def validate(self) -> None:
        n = len(self.y)
        if n == 0 or self.x.ndim != 2 or len(self.x) != n:
            raise ValueError("cohort arrays are empty or misaligned")
        if len(self.case_id) != n or len(self.metadata_index) != n:
            raise ValueError("case identifiers are misaligned")
        if len(set(self.case_id.astype(str))) != n:
            raise ValueError("case identifiers are not unique")
        if not np.isfinite(self.x).all():
            raise ValueError("image features contain non-finite values")
        if np.any((self.y < 0) | (self.y >= len(self.labels))):
            raise ValueError("diagnosis label is out of range")
        if set(self.concepts) != set(self.concept_states):
            raise ValueError("concept arrays and vocabularies differ")
        for name, values in self.concepts.items():
            states = self.concept_states[name]
            if len(values) != n or np.any((values < 0) | (values >= len(states))):
                raise ValueError(f"invalid concept encoding for {name}")

    @property
    def concept_dimension(self) -> int:
        return int(sum(len(states) for states in self.concept_states.values()))

    def gold_features(self, rows: np.ndarray | None = None) -> np.ndarray:
        rows = np.arange(len(self.y)) if rows is None else np.asarray(rows, dtype=int)
        blocks = []
        for name, states in self.concept_states.items():
            values = np.asarray(self.concepts[name])[rows]
            block = np.zeros((len(rows), len(states)), dtype=np.float64)
            block[np.arange(len(rows)), values] = 1.0
            blocks.append(block)
        return np.concatenate(blocks, axis=1)


def load_derm7pt_development(
    feature_archive: str | Path,
    metadata_csv: str | Path,
    official_meta_directory: str | Path,
    *,
    require_development_only_archive: bool = False,
) -> EvidenceCohort:
    """Load official train+valid eligible cases while quarantining official test.

    A legacy source archive containing both development and test embeddings is
    accepted only when ``require_development_only_archive`` is false.  Formal
    post-audit experiments set it true and must supply both a dev-only feature
    archive and a dev-only metadata table with an explicit ``metadata_index``
    column.  That mode prevents the loader from materializing any official-test
    feature, diagnosis, or concept value.
    """

    feature_archive = Path(feature_archive)
    metadata_csv = Path(metadata_csv)
    official_meta_directory = Path(official_meta_directory)
    with np.load(feature_archive, allow_pickle=False) as archive:
        required = {
            "metadata_index",
            "derm_embeddings",
            "metadata_sha256",
            "excluded_metadata_indices",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"feature archive lacks {missing}")
        archive_index = archive["metadata_index"].astype(np.int64)
        metadata_hash = str(archive["metadata_sha256"].item())
        exclusions = tuple(archive["excluded_metadata_indices"].astype(int).tolist())
        archive_role = (
            str(archive["cohort_role"].item())
            if "cohort_role" in archive.files
            else None
        )
    if exclusions != EXPECTED_EXCLUSIONS:
        raise ValueError("frozen result-blind exclusion set changed")
    if len(set(archive_index.tolist())) != len(archive_index):
        raise ValueError("feature archive rows are misaligned")

    split_index: dict[str, np.ndarray] = {}
    for split in ("train", "valid", "test"):
        table = pd.read_csv(official_meta_directory / f"{split}_indexes.csv")
        if list(table.columns) != ["indexes"]:
            raise ValueError(f"official {split} index schema changed")
        split_index[split] = table["indexes"].to_numpy(dtype=np.int64)
    if any(set(split_index[a]) & set(split_index[b]) for a, b in (("train", "valid"), ("train", "test"), ("valid", "test"))):
        raise ValueError("official partitions overlap")

    excluded = set(exclusions)
    development_metadata_index = np.asarray(
        [
            int(index)
            for split in ("train", "valid")
            for index in split_index[split]
            if int(index) not in excluded
        ],
        dtype=np.int64,
    )
    test_metadata_index = {
        int(index) for index in split_index["test"] if int(index) not in excluded
    }
    if len(development_metadata_index) != EXPECTED_DEVELOPMENT:
        raise ValueError("eligible development census changed")
    if len(test_metadata_index) != EXPECTED_TEST_AFTER_EXCLUSION:
        raise ValueError("eligible test census changed")
    if set(development_metadata_index) & test_metadata_index:
        raise ValueError("official test entered the development census")
    if require_development_only_archive:
        if archive_role != DEVELOPMENT_ONLY_ARCHIVE_ROLE:
            raise ValueError("formal loading requires the dev609-only archive role")
        if len(archive_index) != EXPECTED_DEVELOPMENT or set(
            archive_index.tolist()
        ) != set(development_metadata_index.tolist()):
            raise ValueError("formal feature archive is not exactly development-only")
        if set(archive_index.tolist()) & test_metadata_index:
            raise ValueError("formal feature archive contains official-test rows")

    # In strict mode the archive role and complete index census are validated
    # before the embedding member is accessed.  Passing a legacy mixed archive
    # therefore fails without materializing any official-test feature rows.
    if _sha256_file(metadata_csv) != metadata_hash:
        raise ValueError("feature archive and metadata hash differ")
    with np.load(feature_archive, allow_pickle=False) as archive:
        embeddings = archive["derm_embeddings"].astype(np.float64)
    if len(archive_index) != len(embeddings):
        raise ValueError("feature archive rows are misaligned")

    position = {int(index): row for row, index in enumerate(archive_index)}
    try:
        feature_rows = np.asarray([position[int(index)] for index in development_metadata_index])
    except KeyError as error:
        raise ValueError("a development case lacks its frozen embedding") from error
    x = embeddings[feature_rows]
    x /= np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)

    metadata = pd.read_csv(metadata_csv, dtype=str).fillna("")
    if "metadata_index" in metadata.columns:
        parsed_index = pd.to_numeric(metadata["metadata_index"], errors="raise").to_numpy(
            dtype=np.int64
        )
        if len(np.unique(parsed_index)) != len(parsed_index):
            raise ValueError("development metadata indices are duplicated")
        if set(parsed_index.tolist()) != set(development_metadata_index.tolist()):
            raise ValueError("development metadata table has the wrong row census")
        selected = (
            metadata.assign(metadata_index=parsed_index)
            .set_index("metadata_index", drop=False)
            .loc[development_metadata_index]
        )
    else:
        if require_development_only_archive:
            raise ValueError(
                "formal loading requires an indexed development-only metadata table"
            )
        # Backward-compatible path for the already-completed E0/J0 development
        # artifacts.  Formal D1 never enters this all-row metadata path.
        selected = metadata.iloc[development_metadata_index]
    normalized_diagnosis = selected["diagnosis"].str.strip().str.lower()
    unknown_diagnosis = sorted(set(normalized_diagnosis) - set(DIAGNOSIS_GROUPS))
    if unknown_diagnosis:
        raise ValueError(f"unmapped diagnoses: {unknown_diagnosis}")
    grouped_diagnosis = normalized_diagnosis.map(DIAGNOSIS_GROUPS)
    y = (grouped_diagnosis == "MEL").to_numpy(dtype=np.int64)
    if tuple(np.bincount(y, minlength=len(LABELS)).tolist()) != EXPECTED_DEVELOPMENT_CLASS_COUNTS:
        raise ValueError("development diagnosis census changed")

    concepts: dict[str, np.ndarray] = {}
    for name, states in CONCEPT_STATES.items():
        state_to_id = {state: index for index, state in enumerate(states)}
        raw_values = selected[name].astype(str).str.strip().str.lower()
        value_map = CONCEPT_VALUE_MAP[name]
        unknown = sorted(set(raw_values) - set(value_map))
        if unknown:
            raise ValueError(f"development concept vocabulary changed for {name}: {unknown}")
        values = raw_values.map(value_map)
        concepts[name] = values.map(state_to_id).to_numpy(dtype=np.int64)

    reconstructed_score = (
        2 * (concepts["pigment_network"] == CONCEPT_STATES["pigment_network"].index("atypical"))
        + 2 * (concepts["blue_whitish_veil"] == CONCEPT_STATES["blue_whitish_veil"].index("present"))
        + 2 * (concepts["vascular_structures"] == CONCEPT_STATES["vascular_structures"].index("irregular"))
        + (concepts["pigmentation"] == CONCEPT_STATES["pigmentation"].index("irregular"))
        + (concepts["streaks"] == CONCEPT_STATES["streaks"].index("irregular"))
        + (concepts["dots_and_globules"] == CONCEPT_STATES["dots_and_globules"].index("irregular"))
        + (concepts["regression_structures"] == CONCEPT_STATES["regression_structures"].index("present"))
    )
    released_score = selected["seven_point_score"].astype(int).to_numpy()
    if not np.array_equal(reconstructed_score.astype(int), released_score):
        raise ValueError("frozen checklist grouping does not reconstruct seven_point_score")

    cohort = EvidenceCohort(
        dataset_id="DERM7PT-TRAIN-VALID-MEL-EVIDENCE-HANDOFF-DEV609",
        case_id=np.asarray([f"derm7pt:{index:04d}" for index in development_metadata_index]),
        metadata_index=development_metadata_index,
        x=x.astype(np.float64),
        y=y,
        labels=LABELS,
        concepts=concepts,
        concept_states=CONCEPT_STATES,
    )
    cohort.validate()
    return cohort
