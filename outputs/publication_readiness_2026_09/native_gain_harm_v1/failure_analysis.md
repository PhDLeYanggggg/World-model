# Native Cost-Head Failure Taxonomy

2026-09-21. Post-fit descriptive analysis of fixed, design-exposed source results.
No model, seed, threshold or inference input was changed from this readout.

## Confirmed Findings

1. **Conditional harm error persists.** Neural MSE improves global harm fit over
   a constant, yet predicted mean harm is too low in all twelve strict-rule
   scene/seed groups. Underharm4 still underpredicts six of twelve groups. An
   aggregate regression score is not a selected-group safety certificate.
2. **Risk reduction trades away coverage.** MSE/strict switches 2.793% of rows,
   yielding 1.292% ADE gain. Underharm4/strict switches 0.842%, yielding 0.340%.
   Its lower harm is real in this readout, but unequal coverage confounds a
   claim of better ranking. Matched-coverage evidence is currently not_run.
3. **Zero-reference risk differs from positive-easy percentages.** The strict
   MSE and underharm4 rules have positive-easy diagnostic degradation below 2%,
   but still harm exact-zero-CV outcomes. Percentages are undefined for the
   latter. No denominator floor or new pixel allowance is introduced.
4. **One remaining harm does not justify cherry-picking.** Underharm4/strict
   harms one query in deathCircle/seed43; max ADE harm is 0.4503355 pixels. Its
   past path length is 1 pixel, past is not stationary, and all eight available
   neighbor slots contain some past support. Normalized historical velocity
   changes are nonzero. This does not identify an inference-time zero-CV rule:
   exact future CV error is known only for evaluation.
5. **Outcome support remains incomplete.** Underharm4/strict selects 48 unknown
   query/seed outcomes, MSE/strict 283. Unknown rows remain indexed and cannot be
   assigned zero cost. Their risk has not been certified.
6. **No nested-fitting leak was found in this registered dependency chain.**
   Parent, row-scene and outer-scene exclusion checks, normalizer reconstruction,
   score replay and independent errors agree. This removes a specific training
   explanation, not design exposure or annotation uncertainty.

## What Is Not Established

The 3,000-update budget completed, but minibatch loss traces alone do not prove
convergence. For coupa/seed17, both losses spike on the same step2300 batch;
identical sampling and finite checkpoints are verified. Other sites/seeds do
not share a uniformly monotonic trajectory. Arbitrary extra training, a larger
Transformer, or a new loss weight is therefore not yet a diagnosed repair.

The sole remaining zero-CV error may reflect quantized annotations, a short
recent movement or a bad conditional risk estimate. Aggregate past descriptors
alone do not distinguish those explanations. It is not evidence that neighbor
context is missing, nor proof that the task is intrinsically unpredictable.

The four sites remain design-exposed; no calibrated coverage, distribution-free
risk bound or external generalization claim follows from this experiment.

## Next Decision

First separate ranking quality from simple abstention at matched coverage,
using frozen scores and outcome-blind selection. Keep all seeds and report
zero-reference harm, positive-easy harm, gain and unknown support together.
Then register an easy-sensitive conditional-cost objective on the clean nested
views if the diagnosis supports it. Do not use the failing row's future state
as a routing feature or remove it from evaluation. No deployment, Stage5C or SMC.
