# Failure Taxonomy

## Confirmed Findings

1. **Objective tradeoff.** The matched loss reduces positive harm but lowers
   overall and hard ADE performance in all three seeds. All nine fits also
   have a higher native ADE term on the last ten matched logged minibatches.
   Those minibatches are not a full training-loss evaluation.
2. **Reference mismatch.** The training-selected baseline is damping .90 for
   two folds and OLS4 for one. The frozen intervention floor uses a causal
   CV/damping .97 controller. Lower harm to one reference is not certification
   against the other. The loss is an averaged penalty, not the required
   recording-level relative-risk constraint.
3. **Output coverage.** The fold-2 bounded correction cannot exactly represent
   CV on 43,743 excluded-source rows. This is a genuine geometric restriction,
   not an inference-input leakage issue. It does not explain every failure.
4. **Residual support failure.** The label-aware fixed-action oracle improves
   from 146/148 to 147/148 feasible views, but one 082/51 view remains invalid.
   CV is geometrically representable on all 78 rows in that view. A bound-only
   explanation for this particular recording is ruled out.
5. **Not an unlucky seed.** All three current-family seeds lose overall ADE;
   the 3,000-draw locality interval is below zero. Eight of twelve localities
   regress. This remains conditional development evidence, not a new test.

## What Is Still Unresolved

The present evidence does not separate finite model capacity, optimization,
ambiguity of observed motion, label noise and missing scene context on the
remaining difficult recording. The geometric oracle uses future labels for
diagnosis and does not show that its trajectory is causally learnable.

The earlier-family interval spans zero and must not be used as an independent
replication of the current-family result. The two families share development
data and implement different producers. No selected checkpoint or favorable
source replaces either primary comparison.

## Repair Priority

First test a causal reference-complete output parameterization, keeping data,
training budget and loss fixed. Then align newly trained candidate-cost heads
with the actual policy floor and original recording-level risk rules. Only
after that is independent calibration meaningful. Do not raise the 2% budget,
delete the remaining recording, treat missing outcomes as zero, or declare an
empty matched comparison safe.

All repairs above are prospective. This comparison does not deploy a new
model. Stage5C/SMC remain off; image-pixel/raw-frame claims only.
