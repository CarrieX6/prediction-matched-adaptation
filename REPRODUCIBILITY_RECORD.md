# Reproducibility record — 14 October 2026

This record preserves implementation details relocated from the supplement. It complements the [project overview](README.md), [reproduction guide](REPRODUCTION.md), and frozen protocols; it does not replace the scientific fitting scopes, budgets, seeds, or preprocessing specified in the paper.

## Recorded environments

| Scope | Recorded software |
|---|---|
| Original downstream analyses and independent reference-head replay | Python 3.11.9; NumPy 1.26.4; SciPy 1.14.0; scikit-learn 1.5.0; PyTorch 2.1.0+cu121; threadpoolctl 3.5.0 |
| Additional upstream extraction dependencies | torchvision 0.16.0+cu121; timm 1.0.7; pydicom 3.0.2 |
| Finite score controls, repeated partitions, head-size and primary external checks | Python 3.11.7; NumPy 1.24.4; SciPy 1.10.1; scikit-learn 1.2.2; pandas 1.5.3; threadpoolctl 3.5.0 |

Numerical-library thread counts are one. Separate requirement files and environment records retain the environments used for the original and additional analyses; they should not be treated as one merged environment.

## Derm7pt deterministic seed derivation

For original Analysis 1, join base seed `2026090829` and the scope identifiers with vertical bars, encode as UTF-8, calculate SHA-256, read its first four bytes as a little-endian integer, and mask with `0x7FFFFFFF`. Outer scope identifiers are `outer`, repetition; inner diagnostic scope identifiers are `strict_meta`, repetition, outer fold. Identifiers are zero-based and match the archived panels. Direct-feature baselines reuse these partitions. The paper retains the scientific seeds for each bootstrap family and additional experiment; this implementation derivation applies to the stated original fold construction only.

## Frozen checkpoint identity

Official OpenAI CLIP ViT-B/16 JIT checkpoint SHA-256:

`5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f`

This records checkpoint identity, not equivalence between the pretraining configurations of CLIP and BiomedCLIP.

## Independent reference-head verification

Entry point: `upstream_snapshot/scripts/verify_reference_head_20261014.py` in the code archive. Its protocol is under `upstream_snapshot/paper_b_readiness/REVIEW_RECHECK_20261014/`. Reconstruction requires the original, nonredistributed `results/deadline_nested_20260912/<outer_unit>/` panels, selection records, stored heads and predictions under the supplied `--root` directory.

For each of the 30 original outer units, fresh balanced L2 logistic fits compare six inverse-penalty values on four archived inner folds, choose mean class-equal NLL, and refit the selected reference head on all outer-training reference annotations. Reference evaluation uses held-out reference annotations; substitution evaluation uses held-out predicted scores with the same head. Evaluation labels are used for evaluation, not selection. The replay performs 720 inner fits and 30 final fits. Coefficients, intercepts and inner scores match the records; the maximum held-out probability discrepancy is 1.1102230246251565e-16, below the prespecified 1e-7 tolerance. This verifies the existing R procedure and does not add a scientific arm.

The manuscript provenance includes only unit-level discrepancies, counts, selected hyperparameters and source checksums, with no case identifiers, predictions or coefficient arrays. The code archive contains the script and protocol, not the verification result or clinical panels.

## Historical figure-preparation record

During manuscript preparation, OpenAI Codex assisted with drafting, language editing, organization, code preparation, and explanatory-schematic preparation. An image-generation tool supplied a layout reference for an earlier explanatory schematic; figure numbering subsequently changed. The scientific content, equations and connections were implemented and checked in editable vector and presentation files. Figure scripts map stored aggregate estimates, paired interval endpoints and descriptive coefficient ranges to editable vector elements; these preparation materials are not included in this code-only repository. AI was not used to generate or alter medical-image pixels. The submitted manuscript states: “The authors used OpenAI Codex for manuscript polishing and code assistance. The authors reviewed and edited the assisted content and take full responsibility for the final manuscript.”
