# Residual Dynamic-Range Experiment

## Decision Before Fits

The frozen-pool ceiling is only 1.73% on the unchanged primary. Training gains
are also small. In the already exposed training folds, static-history residual
coordinates reach 395--2,578 past-normalized units, whereas common moving-history
corrections are much smaller. These are label-side diagnostics, not features.
This motivates a controlled optimization test before inventing another gate.

Hypothesis: a linear residual readout trained with log1p(ADE) fails to learn large
state-change displacements because both output reach and large-error gradients
are unfavorable. This is not a claim that input information is sufficient.

## Fixed 2x2 Comparison

- `linear_log`: original linear output and log1p per-row ADE; 18 exact-replayed controls.
- `sinh_log`: only readout changes to sinh(clamp(z,-12,12)).
- `linear_asinh`: only loss changes to coordinatewise SmoothL1 between asinh
  predicted residual and asinh target residual, beta=1.
- `sinh_asinh`: both factors change. Targets are never clipped.

The numerical cap is fixed before fits, exceeds all inspected training residual
coordinates, and is not a physical validity or safety bound. Saturations must be
reported. At initialization all arms are exactly CV; the sinh derivative is one.
No extra inference inputs, target-dependent scales, future features or gates.

Same two feature variants (quality-control, directed observed image motion),
three seeds 17/29/43, three grouped fit-scene folds, row-uniform batches, 4,000
updates, AdamW settings, and model size as registered original controls. This
gives 54 fresh fits and 18 reused controls. Fixed final checkpoint, no held-fold
model/epoch/threshold selection. A partial first fit is a runtime pilot only;
resume it without opening held labels. No early success stopping or grid expansion.

All 11,966 fit rows remain. Normalization is learned on training inputs only.
Targets enter supervised loss and later evaluation only. Main endpoint stays
past-normalized ADE with equal physical-scene aggregation; easy limit remains
2%. Report train versus held errors, static-start/stay/moving slices, native
diagnostics separately by recording, saturation, losses, time, exact replay and
resume. Three-scene intervals are descriptive, not independent confirmation.

The asinh objective is a deliberate training surrogate. Its numerical loss is
not comparable to log1p loss and does not redefine the approved scientific
endpoint. Improvement must occur in original ADE and preserve easy cases.

## Boundaries

Offline annotated history may contain retrospective interpolation. Earlier
model selection has exposed these scenes, so no result is called sealed test.
Students/development/calibration/confirmation stay closed. No new source role,
SDD admission, policy promotion, metric/seconds, true-3D, foundation, Stage5C or
SMC claim. This is an optimization diagnosis, not architectural novelty.
If only fit error improves, report overfit/input-support limits, not success.
