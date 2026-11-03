# Interpretation And Remaining Failures

## What Changed The Next Action

The registered detector-history ablation is positive: held-SOURCE AUROC rises
from0.602327 to0.706249; paired delta0.103922 [0.059967,0.162186]. AUPRC rises
from0.079753 to0.144023, against equal-locality unknown prevalence0.041739.
Past inputs therefore contain some transferable signal about the release-label
process. This is not a zero-signal or purely unknowable event on these data.

The target is future-label support, not future motion or physical occlusion.
The positive result is a forest/past-detector component result, not JEPA lift,
Transformer dynamics, a new architecture, independent confirmation or deployment.

## Why The Signal Is Not Yet A Safety Head

Calibration still shifts. Geometry+detector ECE is0.029425 on average but
0.145512 at locality082. Its unknown prevalence there is20.51%. In the failing
recording51, several fold1 models give unknown-row mean scores near3%, despite
16unknown/78target windows. Ranking improves markedly, but the probability
level remains too low for a conservative2%error-risk budget.

Locality020 has AUROC0.4491; locality112 gains essentially nothing from detector
history. Fitting AUROC is0.7873-0.9564, whereas held-locality means range
0.4491-0.8126. The fitting/held differences mix data populations and do not
isolate one cause. Do not drop these localities or call the probe uniformly
generalizing. Every original recording remains, including one-class groups with
undefined AUROC/AUPRC rather than artificially perfect scores.

## Identified Supervision Mismatch

Previous critics supervise actual masked costs only where a future label exists.
At evaluation, wholly unknown rows instead contribute a conservative path
disagreement radius to all/easy positive-harm upper bounds, with no known
reference contribution. These are different training/evaluation targets.
The amplitude study shows that this distinction matters even near the floor.
Known easy harm also fails at useful amplitudes, so missing labels are not the
sole cause and a missingness classifier alone cannot establish compliance.

The next tested hypothesis should compare ordinary known-only costs with
explicit moment-contribution targets for this existing bound. For unknown rows:
benefit lower0,harm upperD,known reference0,known easy reference0,easy harm upperD.
The zeros mean absent *observed denominator contributions*, not zero actual
future errors. D is causal candidate/floor rollout disagreement. No fabricated
future coordinates are used. Partial-known rows keep the original masked-cost
semantics; no claim of identifying their missing future steps follows.

The target constructor now exists in `src/world_model/m3w_supported_bound_targets.py`
with two passing tests. It is not connected to a newly trained critic yet.
Keep `observed_future`, `target_defined` and `unknown_bound_supervision` separate
in training logs: supervising a conservative bound must never be counted as
an observed trajectory label. Future masks stay outside inference features.

## Required Next Evidence

Use the stronger frozen EqMotion proposal and a matched known-only/bound-aware
critic comparison under identical preprocessing, actual seeds and full budgets.
Keep evaluation localities excluded from both forecaster and critic fitting;
retain the original SOURCE roster and the original per-record2%contract.
Evaluate useful amplitude, not only binary100%micro-intervention counts or
near-zero gain. Keep private calibration and independent final roles closed
until a full policy is frozen. Do not retrospectively reuse this probe's held
scores to select a classifier threshold or declare the repair complete.

The broader M3W/A-conference objective remains unmet: independent method-level
efficacy, complete policy risk, composition-conflict contribution and anonymous
cold-start reproduction still need evidence. Pixel/raw-frame/silver only;
Stage5C and SMC remain off.
