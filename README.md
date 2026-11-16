# Lightweight Prediction-Matched Adaptation of Diagnostic Heads for Attribute Mismatch in Medical Image Diagnosis

**Zhuofan Xie, Jinliang Lin, Zishan Lin, Li Chen, Mingyuan Shao, Jie Qi, and Shuo Li**

Manuscript submitted to *Neurocomputing*.

This repository provides the implementation and generated-data checks for the submitted manuscript.

## Why adapt the diagnostic head?

Concept bottleneck models (CBMs) predict clinical attributes before making a diagnosis, making intermediate findings inspectable. An image encoder extracts features, attribute predictors estimate clinical attribute scores, and a diagnostic head combines those scores into a malignancy probability. Here, the diagnostic head is logistic regression.

A head fitted to recorded clinical annotations (reference attributes) receives uncertain predicted scores at deployment. This **reference-to-prediction attribute mismatch** can degrade diagnostic probabilities. Scalar calibration adjusts the head's combined score, but cannot distinguish different attribute vectors that produce the same original logit.

## Prediction-matched refitting

**Prediction-matched refitting (M)** adapts the diagnostic head by relearning only one weight per predicted attribute coordinate and an intercept:

```text
Image → frozen image encoder → fixed attribute predictors → predicted scores s
                                                               ↓
                                         refitted logistic head: σ(w_Mᵀs + b_M)
                                                               ↓
                                                   malignancy probability
```

The attribute vocabulary and upstream predictions stay fixed during head adaptation. With d score coordinates, M refits only **d + 1 diagnostic parameters: 11 on CBIS-DDSM and 20 on Derm7pt**. Unlike scalar calibration, M can change the relative weights of individual attribute scores.

For head fitting, each training case receives predicted scores from an attribute predictor fitted without that case or its group. Nested sample-out evaluation excludes evaluation cases from fitting and model selection at both stages. The upstream models are fitted within the required training scopes, and each head comparison uses identical upstream predictions.

Training a head on predicted attributes follows the established sequential CBM principle. Our contribution is to study this as lightweight adaptation of an existing attribute-based system, compare full-vector refitting with flexible scalar calibration under matched predictions, and identify when the gain holds.

## Main results

The development comparisons use frozen BiomedCLIP image features. Class-equal negative log-likelihood (NLL) gives equal weight to the two diagnostic classes; lower is better.

| Development cohort | Cases | Predicted-score coordinates | Refitted parameters | NLL reduction versus spline calibration |
|---|---:|---:|---:|---:|
| CBIS-DDSM | 655 | 10 | 11 | 0.024 nats |
| Derm7pt | 609 | 19 | 20 | 0.013 nats |

Both paired 95% confidence intervals favor refitting. Sensitivity analyses examine score representation, head capacity, upstream selection, and encoder choice. Improved attribute prediction did not consistently lower refitted-head diagnostic NLL, supporting joint evaluation of predicted findings and their diagnostic combination.

Applied unchanged to 200 external PH² cases, M yields lower natural-frequency NLL and Brier score than spline calibration. The class-equal NLL difference is inconclusive, and spline calibration attains higher AUC. Thus, the development-cohort advantage does not imply improvement in every metric or transfer setting.

## Procedures compared

| Procedure | Diagnostic input and fitting |
|---|---|
| R: reference-input benchmark | Fit and evaluate a logistic head on reference annotations; unavailable at deployment. |
| S: direct substitution | Apply the reference-trained head to predicted scores without changing its weights. |
| C: affine calibration | Fit a positive-slope affine correction to the original predicted-input logit. |
| Cflex: spline calibration | Fit a flexible scalar correction to the same logit. |
| M: prediction-matched refitting | Fit new logistic weights and an intercept on the full predicted-score vector. |

## Getting started

Use an isolated Python 3.11 environment. The original preparation environment used PyTorch 2.1.0+cu121 and torchvision 0.16.0+cu121; choose matching wheels for your platform. Detailed environment records, including the separate environment for additional analyses, are in the [reproduction guide](REPRODUCTION.md) and [reproducibility record](REPRODUCIBILITY_RECORD.md).

```sh
python -m pip install -r requirements.txt

# List the generated-data checks without running them.
python run_synthetic_checks.py

# Run only the six listed generated-data suites.
python run_synthetic_checks.py --execute
```

The six suites contained 139 passing checks in the recorded preparation environment. Four additional checks requiring original project binding artifacts are excluded. These checks validate implementation behavior on generated data; they do not reproduce the clinical experiments.

## Code and reproduction

- [`upstream_snapshot/src/`](upstream_snapshot/src/): attribute generation, diagnostic estimators, calibration, and preprocessing.
- [`upstream_snapshot/scripts/`](upstream_snapshot/scripts/): nested comparisons, aggregation, sensitivity analyses, and feature extraction.
- [Reproduction guide](REPRODUCTION.md): analysis-to-script mapping, frozen configurations, execution requirements, and historical validation scope.
- [Reproducibility record](REPRODUCIBILITY_RECORD.md): software versions, deterministic seed derivation, checkpoint identity, and independent reference-head verification.
- [`MANIFEST_SHA256.json`](MANIFEST_SHA256.json): checksums for the distributed files, excluding the manifest itself; paths retain the original `Paper_B_Review_Code/` archive prefix.

Original implementation filenames and frozen protocols are retained for traceability.

## Data availability

Obtain **CBIS-DDSM**, **Derm7pt**, and **PH²** from their original providers under the applicable access and reuse terms, using the dataset references and cohort definitions in the manuscript.

This code-only repository does not distribute clinical images, case-level predictions, fitted models, feature caches, or clinical result tables. Full clinical regeneration also requires the original split and lineage records and other inputs listed in the [reproduction guide](REPRODUCTION.md#requirements-for-full-clinical-regeneration). Exact clinical regeneration from this repository alone has not been verified.
