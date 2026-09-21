# Cost-Head Failure Starts Before Development Transfer

Completed 2026-09-21. This is a fresh readout of frozen heads on hash-verified
fit arrays, not new training, independent calibration or a deployment result.
The original primary metric and all policy thresholds remain unchanged. The
primary-metric amendment is still pending. Stage5C and SMC remain disabled.

## What Was Replayed

Twelve heads: Transformer/EqMotion x seeds17/29/43 x ridge/neural-cost. Each was
fitted to 11,966 out-of-fold trajectory-cost rows and 306 features. The five fit
recordings are ETH/ETH, ETH/Hotel and UCY/Zara01/02/03. OOF excludes the target
fold from the **trajectory producer**, not from the final cost head: the cost
head was fitted on all these rows. The results below are therefore in-sample
cost-head diagnostics, never held-out risk calibration or transfer results.

All heads, policies and per-recording slices are retained in
[complete tables](complete_results.md) and [machine-readable slices](all_fit_slices.csv).
There are 24 fixed per-agent eligibility comparisons, not 24 independently
trained policies or new scene-level solver evaluations. Membership depends only
on predicted scores and original rules. Actual benefit/harm is used afterwards
for diagnosis, never as an inference feature.

## Main Finding

**The heads can fit global squared error substantially better than a constant
while still getting the sign of expected benefit wrong where switching is
allowed.** This occurs inside their fitting sample and cannot be explained
solely by the later development-domain shift.

| Check | Result |
| --- | --- |
| Harm MSE beats the fit-label mean reference | 12/12 heads |
| Mean predicted harm exceeds mean realized harm over all fit rows | 9/12 heads |
| Mean predicted harm is below realized harm in the fixed eligible subset | 24/24 head-policy combinations |
| Eligible subset has negative realized mean net gain | 23/24 combinations |
| Ridge raw harm outputs below zero | 10.58-50.75% of fit rows, depending on seed/family |
| Actually harmed among those zero-projected rows | 65.37-74.33% |
| Largest 1% of harm labels contributes to squared harm-label mass | 56.23-98.72% |

Thus increasing every prediction by a single factor to repair the overall mean
would address the wrong diagnostic in many heads: global means are already
conservative while the eligible conditional subset is not. This does not prove
any particular replacement estimator will work, nor justify tuning that factor
on the inspected outcomes.

## Concrete Contrasts

Costs below retain the existing past-normalized ADE; they are not probabilities,
meters or raw coordinate errors. Net gain is benefit minus harm, so positive is
favorable. Eligibility is not the final scene-solver selection.

- Transformer seed17/ridge: whole-fit harm MSE 0.96694 versus 39.43806 for the
  mean-label reference (see precise table). The 2,055 conservative-eligible rows
  predict harm 0.003926 and net gain +0.10845, but realize harm 0.11120 and net
  gain -0.09479.
- EqMotion seed17/ridge: 1,965 conservative-eligible rows predict harm 0.000460,
  while actual mean harm is 0.30251. Predicted net gain +0.09695 becomes -0.28083.
- EqMotion seed29/neural-cost: 170 conservative-eligible rows predict harm
  0.02151 and net gain +0.09810, versus realized harm 0.89220 and gain -0.85240.
  Softplus outputs are already nonnegative, so ridge's clipping is not the only
  failure mechanism.
- EqMotion seed17/neural-cost/conservative is the retained positive exception:
  44 fitting rows have mean gain +0.000941, versus predicted +0.06112. This small
  in-sample subset is not selected as a winner and supplies no deployment or
  independent calibration claim. Its moderate counterpart is negative.

## What Was And Was Not Wrong

1. **No discovered ordering or scale corruption in the checked path.** All 12
   saved normalizers exactly match their original fit arrays. Frozen code,
   producer identities, OOF-fold exclusion, feature identity, target-group
   hashes and checkpoints pass binding checks. Replaying fixed original
   first/middle/last batches in all 18 producer folds reproduces 5,748 feature
   rows and independently recomputed two-column targets exactly. This is a
   representative raw replay, not a claim to reconstruct every raw training row.
2. **Zero-projected ridge cost is not evidence of zero risk.** Ordinary linear
   regression emits negative costs; the existing readout projects them to zero.
   Many such rows are harmed. Projection itself raises negative values, so
   simply removing it would not repair safety. It exposes inaccurate estimates
   near the selection boundary, not a reversed mathematical clipping operation.
3. **Global fit quality misses the selected region.** Mean squared error and
   global mean agreement do not establish reliable cost estimates after selecting
   on low harm and high gain. The error exists even without new development data.
   This is a decision-relevant failure, not evidence that regression in general
   is unsuitable.
4. **Target magnitude is concentrated.** Only 120 rows per target population
   account for 56.23-98.72% of squared harm-label mass. This motivates checking
   conditioning and boundary-relevant learning, but it is not the final loss
   contribution or proof that those rows caused failure. Removing them or
   changing their weights would change the learning objective and needs a new
   registered experiment.
5. **Model/optimization/feature causes remain unresolved.** This study does not
   distinguish function-class misspecification, insufficient optimization or
   insufficient causal information. Out-of-fold producer versus full-producer
   shift can still matter at development time, but cannot be the sole explanation
   for a failure already present on head-training rows.

## Consequence For The Next Experiment

The previous global-budget versus conditional-easy mismatch still applies. Better
cost fitting cannot make those two constraints equivalent. The next registered
design needs both a risk target aligned with the approved easy metric and a
readout that reports decision-region reliability, not just overall loss.

After the pending metric decision, the shortest falsifiable training comparison
is a fixed matched-head experiment on legal source data, with separately held
cost-validation groups, explicit treatment of target scale, and the same frozen
trajectory candidates. Test prediction reliability around the intervention
region alongside overall MSE. Distinguish boundary-relevant weighting, which
changes the target risk, from a pure numerical reparameterization. Do not use
the present fitting scores as a substitute for that held-out evidence.

Any deployment claim will additionally need independent calibration/confirmation
support and safe easy-case outcomes. No new loss, threshold or fitting run is
authorized by this diagnosis itself; no model is promoted here. Historical
Stage26/37 results remain exploratory after the lineage/test-use audit. The
project remains a dataset-local trajectory/world-state research track, not
true 3D, metric, seconds-level, foundation or submission-ready success.

## Verification And Limits

CPU Transformer and original-device MPS EqMotion head readouts complete with
workers0 and four compute threads. The work is small because it replays cost
heads, not because a training run was shortened. Completed resume reuses every
seed and makes no new forward calls. Source forecasts are rerun only in the
fixed-batch verification, never to create new development results.

Independent scalar reductions check 48 conditional summaries. Fixed source
replay checks 54 batches/5,748 repeated rows; these are not independent samples.
45 scoped tests pass and one opt-in MPS unit test is skipped in the default suite;
actual MPS head and producer replay is separately completed, not replaced with
CPU or NumPy. Full historical report-writing integrations are not rerun. All
required sessions are terminal. Reproduction and exact hashes are in
[execution notes](execution_notes.md).
