# PH2 transfer-mechanism analysis protocol

## Status and purpose

This is a post-result, descriptive analysis specified after the PH2 transfer
performance was known. It does not fit, select, recalibrate, or alter any
diagnostic model. Its purpose is to locate observable changes at the predicted
concept interface that accompany the external performance result. The analysis
cannot identify a causal mechanism because PH2 has no harmonized seven-point
concept annotations.

## Frozen inputs

- The 25 archived Derm7pt BiomedCLIP outer models and their original held-out
  concept-score panels.
- The 200 PH2 BiomedCLIP feature vectors and the recovered parameters of those
  same 25 concept predictors.
- The archived annotation-trained reference heads used by substitution S.
- The original Derm7pt and PH2 diagnosis labels, used only for stratification
  and descriptive checks.

No PH2 observation enters fitting, model choice, normalization, or threshold
selection.

## Analysis populations

The primary analysis uses one record per case. Each Derm7pt case has five
out-of-fold concept-score vectors, one from each repetition; these are averaged
within case. Each PH2 case has 25 score vectors, one from each fixed Derm7pt
source model; these are averaged within case. The corresponding reference
logit is averaged on the logit scale. Analyses are reported for all cases and
separately for diagnosis class 0 and class 1. The class-specific analyses
address the narrower PH2 negative class and other differences in case mix.

A same-model sensitivity analysis compares each source model's Derm7pt outer
holdout with its predictions on all 200 PH2 cases. The 25 resulting values are
reported as a median and full range. They are not treated as independent
replicates because training sets overlap and the same PH2 cases recur.

## Prespecified summaries

1. **Marginal concept-score shift.** For each of the 19 categorical state
   probabilities, compute the one-dimensional Wasserstein-1 distance between
   Derm7pt and PH2. Summarize the 19 distances by their median and maximum.
   Distances remain on the probability scale.
2. **Dependence shift.** Within each cohort, compute the 19-by-19 Spearman
   correlation matrix. Across the 171 unique off-diagonal pairs, summarize the
   absolute between-cohort correlation change by its root mean square, median,
   and maximum.
3. **Reference-logit shift.** Treat cohort membership (Derm7pt=0, PH2=1) as
   the endpoint and compute the AUC of the averaged reference logit z_S. Report
   the signed AUC, its orientation-free value max(AUC, 1-AUC), and the PH2 minus
   Derm7pt mean-logit difference. This is a shift diagnostic, not diagnostic
   performance.

Coordinate-level Wasserstein distances and correlation-pair changes are
retained in full. The largest values are descriptive rankings; they are not
selected hypotheses.

## Uncertainty

For each of the three analysis populations (all, class 0, class 1), 10,000
draws independently resample cases with replacement within Derm7pt and PH2.
The same resampled cases are used for all summaries in a draw. Percentile 95%
intervals use seed 20261011 and condition on the archived fitted models. The
intervals are not adjusted for the number of coordinates or correlations.

## Interpretation rule

Large class-specific score or correlation changes locate an interface shift
that is not explained solely by diagnosis prevalence. Cohort separability of
z_S locates a shift after the fixed reference head compresses the score vector.
These observations can support an association between interface shift and the
external result. They cannot establish that the measured shift caused the
performance change or measure external concept accuracy.
