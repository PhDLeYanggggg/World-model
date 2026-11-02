# Better Cost Fit Does Not Identify A Safe Useful Policy

## Completed Controlled Readout

All 36 real Torch fits and two exact full-budget refits finish. Both new arms
share capacity, causal inputs, initialization, draws and budgets. All 72 original
SOURCE groups remain. No thresholds, labels, role assignments or forecasts are
changed after readout. See the separate verification receipt for numerical
replay; this analysis does not substitute for it or for independent replication.

The identified arm loses 4.079383 percentage points of known-label utility
relative to its matched marginal arm, conditional locality-bootstrap 95% CI
[-5.548596,-2.623124]. The registered completion-lower effect is -5.321489 pp
[-7.088381,-3.692526]. All twelve locality-averaged known effects are negative,
as are all three cost-seed effects. Even the completion-upper paired interval
is negative: -2.837276 pp [-4.277800,-1.369162]. Missing outcomes are not the
sole explanation for this failure. Required empty count-matched comparisons
remain not estimable, not safe zero.

## What Improved

|Quantity|Marginal supervision|Occurrence and conditional supervision|
|---|---:|---:|
|Fixed fitting anchor loss|0.115738|0.108436|
|Fixed fitting occurrence BCE|0.366042|0.149985|
|Fixed fitting conditional magnitude MSE|0.127004|0.098434|
|Known-row occurrence Brier|0.109240|0.037867|
|Cost-scale normalized B/H/R/ER/EH MSE|0.072576/0.212650/0.631342/0.001203/0.007582|0.066884/0.181976/0.539928/0.001046/0.006306|
|Known unsafe action occurrences admitted|265,601|139,550|
|Required recording-risk views supported|15/444|55/444|
|Known all/easy positive-risk violations|181/293|71/150|
|Worst known easy net degradation|6.250110%|1.264269%|

All five aggregate normalized cost MSE point estimates improve. The occurrence
and conditional targets are learnable on this fitting cohort; this is not a
runtime failure or evidence that the labels have no signal. The paired diagnostic
intervals are retained in `diagnostic_effects.json`. They are SOURCE diagnostics,
not independent calibration. Fixed monitor losses are not validation losses.

## What Was Lost

Known ADE gain over the same floor falls from 6.909619% to 2.830236%, below the
frozen guarded TDEF's 11.890200%. Switching falls from 35.837402% to 17.934363%
of repeated row views. Rejected known safe easy action occurrences increase
389,742 to 524,628; rejected strictly beneficial safe easy occurrences increase
364,680 to 492,385. Known-easy-support blocked views rise 10 to 26, and undefined
selected-easy record views rise 15 to 38. The singleton joint-support obstruction
remains in five repeated views. These are repeated action/context/seed counts,
not independent samples or all actual selected switches.

Neural-action use falls from 84,631 to 51,224 repeated rows. Both policies use
the same forecast paths; this is a routing change, not a neural dynamics lift.
No path-use ablation attributes this result to JEPA or Transformer features.

## Why Better Regression Is Not Enough

The new factorization removes the preceding cost-tied BCE semantic problem,
but does not solve threshold-sensitive selection. The pointwise easy condition
depends on H1-0.02R1 whenever q>0. Better occurrence Brier alone cannot change
that sign. Lower aggregate cost MSE also does not establish that small relative
errors near this boundary have the correct sign in every selected recording.

The measured mechanism is increased conservatism: fewer unsafe admissions,
but many more beneficial rejections and lost denominator anchors. This is a
direct decision-level measurement, not a unique causal attribution of every
finite-network error to one loss term. Shared features, conditional weighting,
limited context and support restrictions can still matter; they were not
separately randomized in this experiment.

Easy net preservation must not be substituted for positive-harm safety. The
identified policy has no known easy net-degradation view above 2%, yet 150 easy
positive-risk violations remain. Gains on different easy rows can offset harm
in the net metric; the approved risk contract does not allow that cancellation.
Only 55/444 required record-risk views are supported, and undefined denominators
remain not estimable. Neither a small intervention rate nor zero interventions
certifies a useful learned-intervention result.

## Decision And Next Evidence

Fitting gates are 6/9. No deployment promotion or independent-data opening.
Do not tune this readout's threshold, promote occurrence calibration into a
safety theorem, or repeat the old hurdle study as a new method. The current
controlled result falsifies the narrow hypothesis that conditional identification
alone repairs this frozen guard's complete-policy utility and risk.

The next useful evidence should localize conditional magnitude errors on the
actual decision boundary and support-deficient queries, then check available
past/context support in already approved training/development roles without
opening a reserved final test. Any resulting loss or allocation change needs
a new isolated, frozen comparison. More fitting accuracy on the same exposed
SOURCE roster cannot establish independent world-model generalization.

This is EuropeanSquares detector-silver local-image obs8/pred12/stride12
raw-frame partial-mask ADE, not SDDt+50, metric/seconds, true3D, foundation,
human-gold, new trajectory dynamics, physical safety or submission readiness.
Three cost seeds share forecaster43. Stage5C and SMC remain off.
