# Capacity Helps Some Decisions, but Does Not Repair Risk Transfer

This is a development experiment on the same12 previously opened European
localities. It changes the capacity of a conditional cost head, not the frozen
motion forecaster. The result cannot establish independent deployment safety.

## What Was Trained

72 single-locality fits, each with two arms: affine logits with1,905 parameters
and a32-wide SiLU head with12,357 parameters. Both share the bounded moment
decoder, initial predictions, causal inputs, training queries,2,000-update budget
and proper squared cost/decision-score losses. No best-checkpoint selection.

Each evaluated fitting locality supplies neither labels nor preprocessing to
its head. The reverse direction is also evaluated. The two outer-held localities
of each view remain excluded; roles rotate across the216 views. Reusing a head
across views does not create additional independent training experiments.

## What Worked

Both arms optimize the registered training objective. The equal-fit mean monitor
loss falls from0.82560 to0.61484 for affine logits and0.50199 for the nonlinear
head. This is a fixed128-query training monitor, not validation loss.

At matched intervention counts, nonlinear selection has0.65629% lower ADE than
affine selection; the nominal12-locality bootstrap interval is
[0.34267%,1.00558%]. All three seed point estimates are positive:
0.79549%,0.64062%,0.53274%. The arms have different admissible pools, so this is
not a pure within-pool ranking effect. It is a selected-set improvement at an
equal count, not a safety guarantee or an independent-test success.

## What Failed

The registered primary, nonlinear minus affine normalized gain/all-risk/easy-risk
MSE, is+0.07709 with nominal interval[-0.10209,0.30087]. Lower is better: the
experiment does not establish improved cost prediction. The nonlinear head also
has worse score MSE than its initial training-prior decoder in the secondary
comparison: +0.19601 [0.04356,0.33860]. This decoder is not a constant forecast:
its benefit/harm outputs still depend on the causal rollout envelope.

The accuracy improvement does not preserve the selected-risk objective:

| Matched policy | All-risk violations / defined | Easy-risk violations / defined | Worst easy ADE degradation | Unknown-label interventions |
|---|---:|---:|---:|---:|
| Affine |158/216|166/216|2.05786%|7,094|
| Nonlinear |168/216|178/216|4.18009%|8,727|

These are dependent directional views, not independent Bernoulli trials.
Unknown-label interventions cannot be declared harmless. Positive-harm ratios
retain the selected-floor denominator and2% budget; net average improvement
cannot cancel positive harm for this estimand.

Without count matching, the nonlinear policy intervenes more often and its
worst easy degradation is9.98191%. The completely unprotected neural forecaster
improves mean all-case ADE by7.73066% but degrades easy ADE by15.52122% on average;
all216 views violate the selected all-risk budget. It is not deployable.

## Failure Taxonomy

1. **Transfer of conditional costs:** lower source-fitting loss does not establish
   lower locality-held score error. This is consistent with overfitting, source
   shift or imperfect conditional targets; this experiment does not isolate
   which mechanism dominates.
2. **Prediction versus decision:** selected-set ADE can improve while cost MSE
   does not. Global squared prediction error and a sparse thresholded action are
   different targets; neither metric substitutes for the other.
3. **Decision versus safety:** even matched counts do not match selected harm.
   Nonlinear selection admits a different set and produces more violating views.
4. **Rare or unobserved outcomes:** the observed-label report excludes unknown
   losses, while interventions on those rows remain counted. A complete-case
   estimate is not a bound on unobserved harm.
5. **Evidence scope:** already exposed localities, overlapping trajectories and
   shared producers make this a developmental comparison. The nominal bootstrap
   does not account for the entire adaptive research history.

The result does not prove causal context is useless. It does show that adding
this capacity under the same proper-loss training is insufficient for the
registered accuracy-and-risk objective. Nor does it justify weakening the2%
budget, selecting the best observed locality or relabeling abstention as safety.

## Next Controlled Question

Before another architecture or threshold sweep, separate source-fitting error
from transfer error near the actual all/easy decision boundary. Use the frozen
predictions to measure signed residual bias, harmful-tail underestimation,
support coverage and unknown-label concentration per source and query. Keep this
descriptive: no policy selection from the directional readout.

The next training change should be chosen from that evidence and registered
before fitting. A train-source-only recording/block validation control could
test whether the fixed long fit adds source-specific confidence that fails to
transfer, without using the other fitting locality as a tuning set. Compare
against the already tested loss/occurrence/capacity controls before spending
another training budget. Do not assume early stopping or shrinking to the prior
will solve safety: even the initial-prior policy is not safe here.

The current deployment remains unchanged. Independent selection, risk calibration
and confirmation are not run. Obs8/pred12 at stride12 raw frames, image-local
detector-silver only; no metric, seconds, human-gold, true3D or foundation claim.
No Stage5C execution and no SMC.
