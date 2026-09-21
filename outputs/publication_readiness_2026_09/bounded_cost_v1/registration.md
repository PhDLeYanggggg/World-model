# Fixed Forecast-Disagreement Cost Parameterization Study

Registered before any new real fit or model readout. Three source-only arms,
four explored physical sites and seeds 17/29/43: 36 fresh cost heads, each 3,000
updates with scene-uniform batches of 256. No new forecaster or threshold search.
Original val/test/main/external/bookstore roles remain closed. All four source
sites are design-exposed; this cannot become independent confirmation.

## Motivation and Fixed Hypothesis

The previous joint module contributes no additional decisions. A separate
geometry audit verifies that full-grid realized benefit plus harm is bounded by
D, the mean distance between the two frozen forecasts. Existing expected costs
sometimes exceed D. On training-only complete outcomes, dividing cost by D also
reduces top-1%-label squared-energy concentration. Neither observation proves
that enforcing the bound will improve intervention or solve exact-zero safety.

Test two distinct changes, not another architecture sweep:

1. `direct_native`: softplus benefit/harm multiplied by the complement-training
   cost scale; ordinary native-cost MSE.
2. `bounded_native`: for the same two positive raw outputs u, predict
   D*u/(1+sum(u)); the same native-cost MSE. This isolates output parameterization
   while holding the training loss, data and sampler fixed.
3. `bounded_fraction`: same bounded output with MSE on cost/D. This deliberately
   changes sample weighting; it is not an identical-objective comparison.

The elementary triangle inequality and bounded regression are not claimed as new
theory. These outputs are continuous costs, not calibrated probabilities.
The bounded sum is a necessary consistency condition, not sufficient accuracy
or protection. No statistical guarantee follows from satisfying it.

## Matched Training

All arms use the same 356 causal features: existing 355 cost features plus
log(1+D). D uses all twelve requested predicted steps and no future-validity mask.
One 64-wide GELU hidden layer, two output logits, identical raw parameter
initialization including a zero final layer, identical draws and 3,000 updates.
AdamW lr0.001, decay0.0001, gradient clip5. Atomic checkpoint every500 steps,
heartbeat every100. CPU4/inter-op1/workers0, arm64 environment.

Only complete future training outcomes supply cost labels. Incomplete labels
remain unknown, never zeros. The same complete support determines all training
normalizers and all three samplers. At D=0, predictions and supervised costs are
structurally zero; the fraction loss contributes zero. No denominator epsilon is
used to alter the scientific outcome or exact-zero evaluation rule.

Frozen two-site-excluded producers supply each training row; the outer source
and row source remain excluded upstream. Each outer forecaster and every other
artifact remain fixed. A separate direct control is freshly trained on the same
complete-only support; the older partially supervised cost head is a reference,
not a matched training arm. This separates support changes from the new head.

## Fixed Readout

Complete every endpoint before computing the new outer scores. Freeze all score
and choice archives before loading the evaluation targets. Evaluate every seed
and arm, without selection, early stopping or best-checkpoint claims. The final
fixed-budget checkpoint is used, not a validation-selected checkpoint.

Three policies per head use only past inputs:

- net-stop: predicted benefit > harm, last past step nonzero, D>0.
- strict-stop: net-stop plus predicted harm <=0.1*predicted benefit.
- matched-count: among non-stop/D>0 proposals, select top predicted net gain at
  exactly the frozen legacy stop-MSE-strict count for the outer source/seed.
  Global query ID breaks ties. This is offline allocation, not an online policy.

Primary contrast: bounded-native strict-stop minus direct-native strict-stop in
equal-physical-scene native ADE gain. The same-count contrast is the mechanism
control against gains from simply abstaining more. Fraction-minus-bounded-native
contrasts identify the weighting change. Full population, partial and unknown
outcomes remain indexed. Report complete zero-CV absolute harms, parent-frozen
training-q25 positive-easy and q75 hard slices, FDE, tails, counts, conditional
cost error and selected unknowns. Use 3,000 physical-scene bootstrap draws after
averaging seed errors; four explored sites are not independent confirmation.

Success requires meaningful paired accuracy/coverage benefit without violating
strict zero-CV protection or the existing <=2% positive-easy diagnostic rule.
Do not declare success from lower global MSE or output-bound compliance alone.
No post-hoc threshold, tolerance, sample exclusion or favorable-seed selection.
Missing-outcome full-grid absolute-gain intervals are supplementary diagnostics,
not replacements for native ADE/FDE or full-population safety certificates.

Local smoke uses the first100 updates of one registered fit; resume completes
that same fit. Slowness is not a reason to change its budget. Code/config drift
requires a new identity. No new deployment, metric/seconds/true3D/foundation,
Stage5C execution or SMC. Independent calibration/confirmation remain not_run.
