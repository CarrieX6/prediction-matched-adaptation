# Reproduction guide

Implementation entry points and execution requirements for **Lightweight Prediction-Matched Adaptation of Diagnostic Heads for Attribute Mismatch in Medical Image Diagnosis**. See the [project overview](README.md) for the method and submitted-manuscript results.

## Layout

- `upstream_snapshot/src/`: attribute generation, diagnostic estimators, calibration, and preprocessing.
- `upstream_snapshot/scripts/`: Analysis 1 scalar/vector comparison and aggregation; Analysis 2 representation/capacity/temperature; Analysis 3 concept-selected upstream replacement; feature extraction.
- `upstream_snapshot/paper_b_readiness/ENCODER_ROBUSTNESS_CLIP_20261010/`: locked protocol and executable entry points for the second frozen-encoder Analysis 1 check.
- `upstream_snapshot/paper_b_readiness/PH2_TRANSFER_MECHANISM_20261011/`: locked protocol for the post-result predicted-interface shift analysis.
- `upstream_snapshot/checks/` and `tests/`: implementation checks on generated data.
- `upstream_snapshot/configs/` and `upstream_snapshot/paper_b_readiness/`: frozen scientific configuration files.
- `run_synthetic_checks.py`: explicit generated-data check allowlist.

Original scientific source files are preserved, including dated names and internal imports. The single `upstream_snapshot/results/deadline_nested_20260912/protocol.json` file is an imported configuration, not an experimental result. It is retained at its original path to avoid changing the implementation.

## Environment and safe checks

Python 3.11 was used. Create an isolated environment and install `requirements.txt`. Install matching PyTorch/torchvision wheels for the local CPU/CUDA platform; the tested environment used PyTorch 2.1.0+cu121 and torchvision 0.16.0+cu121. Do not install a different CUDA runtime merely to view the sources.

```sh
python -m pip install -r requirements.txt
python run_synthetic_checks.py
python run_synthetic_checks.py --execute
```

Running `python run_synthetic_checks.py` without `--execute` lists the six suites and reports `NOT_RUN`. The explicit execution runs only the six named generated-data suites: Cflex (24 checks), P1 unit (22), P1 aggregation (9), P2 (54), P3 upstream (17), and P3 development runner (13): 139 checks passed in the preparation environment. This is not a clinical reproduction or new confirmation experiment. Four P2 tests requiring original project authorization/binding artifacts are excluded and remain NOT_RUN; the test adapter lists their names.

The runner disables GPU access and pytest cache creation. Do not use unrestricted test discovery as a replacement for this allowlist.

## Scientific entry points

All paths below are relative to `upstream_snapshot/`.

| Analysis | Entry point | Frozen configuration |
|---|---|---|
| Analysis 1 generation/refitting | `scripts/run_p1_formal_orchestrator_v7_20260916.py` | `paper_b_readiness/P1_FORMAL_ORCHESTRATION_SPEC_V7_20260916.json`, `P1_PROTOCOL.json` |
| Analysis 1 aggregation | `scripts/run_p1_complete_aggregate_v7_20260916.py` | `paper_b_readiness/P1_AGGREGATE_SPEC_V7_20260916.json` |
| Spline scalar control | `scripts/run_b_cflex_control_20260917.py`, `aggregate_b_cflex_control_20260917.py` | `paper_b_readiness/CFLEX_CONTROL_20260917/PROTOCOL.json` |
| Analysis 2 representation/capacity/temperature | `scripts/run_neurocomputing_p2_scientific_v3_20260916.py` | `paper_b_readiness/P2_PREP/SCIENTIFIC_V3/P2_PROTOCOL.json` |
| Analysis 3 upstream selection | `scripts/run_neurocomputing_p3_dev_20260916.py` | `paper_b_readiness/P3_PREP/EXECUTION_V1/PROTOCOL.json` |
| CBIS feature extraction | `scripts/extract_paper_b_cbis_full_mammogram_biomedclip_v2.py` | `configs/paper_b_cbis_mass_train_embedding_protocol_v2_20260910.json` |
| Derm7pt feature extraction | `scripts/extract_derm7pt_biomedclip_features.py` | command-line inputs and dataset metadata |
| Second frozen encoder (Analysis 1) | `paper_b_readiness/ENCODER_ROBUSTNESS_CLIP_20261010/extract_clip_features.py`, `generate_clip_panels.py`, `run_analysis.py` | `paper_b_readiness/ENCODER_ROBUSTNESS_CLIP_20261010/PROTOCOL.json` |
| Paired encoder concept quality | `paper_b_readiness/ENCODER_ROBUSTNESS_CLIP_20261010/paired_concept_quality.py` | Same outer cases; 10,000 paired case-bootstrap draws, seed 20261012 |
| PH2 predicted-interface shift | `scripts/analyze_ph2_transfer_mechanism_20261011.py` | `paper_b_readiness/PH2_TRANSFER_MECHANISM_20261011/PROTOCOL.md` |

For source inspection and feature-extractor imports, add these directories to PYTHONPATH: `upstream_snapshot/src`, `upstream_snapshot/scripts`, and `upstream_snapshot/vendor/iclr_gate1a_anchored_mnpd_snapshot_20260904`.

## Requirements for full clinical regeneration

1. Obtain CBIS-DDSM and Derm7pt, and PH² for external evaluation, from their original resources under the applicable access/reuse terms. Dataset citations and cohort definitions are in the manuscript.
2. Obtain the exact BiomedCLIP image-encoder checkpoint and preprocessing dependencies. CBIS uses the v2 extraction route: 384 × 256 display resize followed by the 224 × 224 encoder transform. The inherited base extractor has a different default and must not be substituted. This extraction path uses the image encoder through timm, not a text tokenizer.
   The second-encoder check additionally requires the official OpenAI CLIP ViT-B/16 JIT checkpoint; the audited checkpoint SHA-256 is `5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f`. `IMPLEMENTATION_CHECKPOINT.json` distinguishes this executed artifact from the model-family reference recorded in the locked protocol.
3. Reconstruct or obtain the frozen feature matrices, label/attribute encodings, group identities, cohort ordering, and split/lineage manifests referenced in the configurations. These are not included in this code-only archive. The manuscript and supplement describe preprocessing; they do not supply all clinical execution dependencies.
4. Resolve historical absolute paths in a separate local configuration copy and validate dataset identity. Path remapping alone does not establish equality of the underlying data.
5. Supply the original evidence bindings and authorization artifacts required by the protected runners. Do not disable their checks, fabricate approvals, or access a reserved test partition to work around missing dependencies.

Until those dependencies are available, exact clinical regeneration from this archive alone is **not verified**. The code can be inspected and the generated-data checks can run. Figure-generation materials and clinical result archives are not distributed in this repository.

Third-party dependencies retain their original licenses.

## Added entry points in the 13 October 2026 revision

Paths below are relative to `upstream_snapshot/`.

| Analysis | Entry point |
|---|---|
| Direct-feature baselines and coefficient replay | `scripts/run_added_baselines_20260930.py` |
| Additional paired metric summaries | `scripts/summarize_added_baselines_20260930.py` |
| PH² audit, feature extraction and external evaluation | `scripts/run_ph2_external_20260930.py` |
| Matched five-model, 12-coordinate permutation null | `scripts/analyze_interface_shift_null_20261012.py` |
| Mahalanobis associations and coordinate contributions | `scripts/analyze_ph2_failure_connection_20261012.py` |
| Paired encoder gain differences | `scripts/run_encoder_delta_bootstrap_20261012.py` and local helper `paper_b_readiness/ENCODER_DELTA_20261012/encoder_delta_bootstrap.py` |
| Two-to-five-coefficient spline sensitivity | `scripts/run_cflex_df_sensitivity_20261013.py` |
| Sixteen-test Holm correction and class-wise associations | `scripts/audit_revision_statistics_20261013.py` |

The earlier 19-coordinate unmatched shift summaries are descriptive historical outputs; manuscript inference uses the matched permutation runner. The class-wise bootstrap uses fixed Derm7pt covariance, matched five-model probability outputs and seed 20261025. The encoder gain-difference runner loads its helper from this archive, not from an attachment directory.

Clinical execution additionally needs the licensed datasets, original split manifests, archived outer/inner predictions and heads, checkpoint/feature/label caches, and access receipts referenced by the scripts. Original paths are preserved for traceability; configure them for the receiving environment before execution. The code-only archive excludes these case-level inputs. Use the paired metric definitions, recorded aggregation rules and seed families rather than pooling distinct analyses. No new training or unrestricted clinical execution was performed for archive validation.

## Bounded experimental revision (code v6)

`upstream_snapshot/scripts/run_bounded_revision_20261013.py` executes the completed finite source-side checks. Its modes are `controls` (PCA, random projections, standardized M and L1), `stability` (two new full nested CBIS partitions and head-only training sizes), `ph2` (primary 25-model class-wise associations, rank accounting and per-model logit decomposition), and `summarize` (paired conditional intervals and coefficient summaries). The frozen protocol and the implementation note are in `upstream_snapshot/paper_b_readiness/BOUNDED_REVISION_20261013/`. No outcome-dependent further mode or external adaptation was added.

This runner's executed environment was Python 3.11.7 with NumPy 1.24.4, SciPy 1.10.1, pandas 1.5.3, scikit-learn 1.2.2 and threadpoolctl 3.5.0, using one numerical thread. Install `requirements_bounded_revision.txt` in a separate environment for this runner; the original requirements file remains the specification for earlier preparation environments.

All new partitions refit the upstream and diagnostic stages under patient exclusions. The original partition is retained from its archive. Score controls use the existing nested feature/prediction panels with direction fitting restricted to training features. PCA has 10 CBIS or 12 Derm7pt dimensions. All three fixed random seeds 20261025/26/27 are reported. Standardization is fitted on training rows, and all diagnostic candidates use the same six-value penalty grid and four inner folds. L1 uses liblinear, which also penalizes the intercept; categorical weights are block-centered only for interpretation, after sparsity counts.

The head-only 50%/75% checks fix upstream predictions, archived reference logits, and full training-side spline knots/basis. Both diagnostic heads use the same retained patients; inner validation and outer evaluation rows remain fixed. The 100% result is the original archived result. This is not an upstream or complete-pipeline learning curve. A source-head replay under a different dependency environment changed a near-tied spline candidate; `implementation_note.json` records this, and the original main results were preserved rather than replaced by the replay.

The primary PH² connection now uses all 25 source-model probabilities. Its source covariance uses the archived OOF five-repeat mean scores; differing averaging depths make the resulting associations descriptive. The matched-five permutation and association runners listed above remain separate sensitivity analyses. Per-model M-minus-reference logit contributions, then their model average, are not a decomposition of Cflex or of an ensemble probability. New conditional paired bootstrap summaries use seed 20261034, with metrics averaged across repeated partitions rather than probabilities.

Execution requires inputs excluded from this code-only archive: the archived nested panels and heads under `results/deadline_nested_20260912/`, the primary Cflex outer/inner archive, cohort feature/label/group arrays and configuration, PH² interface-analysis input arrays, and per-model PH² probability arrays. The runner preserves historical path conventions; remap in a local execution copy and verify identity and lineage before regeneration. It does not generate images or change an external test model. `--help` only parses the interface and imports dependencies; delivery validation did not rerun experiments or the historical 139 generated-data checks.


## 14 October reference-head verification

`upstream_snapshot/scripts/verify_reference_head_20261014.py` independently refits the existing reference heads using their archived permitted training/validation panels. Its frozen protocol is under `upstream_snapshot/paper_b_readiness/REVIEW_RECHECK_20261014/`. The executed replay verifies all 30 units, 720 inner fits and 30 final fits; no new arm is added. Verification results accompany the manuscript provenance, not this code-only archive.

Run the script with `--root` pointing to the reconstruction root containing the legitimately held original `results/deadline_nested_20260912/<outer_unit>/` input records. It requires the original downstream environment and the nonredistributed panels, selection/head records and predictions. See [REPRODUCIBILITY_RECORD.md](REPRODUCIBILITY_RECORD.md) for versions, deterministic seed derivation, checkpoint identity and precise replay scope. Clinical datasets and panels are not bundled.
