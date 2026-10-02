# Targeted Harm-Head Learning, Not Another Threshold Search

The source experiment establishes useful past-quality information; the component
diagnosis identifies total/easy-harm corrections as the main unsafe expansion
mechanism. Benefit and reference corrections are not the dominant average cause.

Next register one targeted training comparison: keep the original benefit and
both reference estimates frozen, and learn only total/easy-harm from the same
seven past-quality features. Compare the unchanged original, a two-harm additive
control, and a positive-link conditional harm head. Preserve each frozen leaf's
weighted TRAIN mean through an explicit training-only normalizer, rather than
shrinking all predictions or calibrating a global intercept on validation.

A candidate positive-link fit uses an exponential link and weighted conditional
deviance with fixed regularization, rather than permitting negative raw harm and
clipping it to zero. Zero-harm training leaves require an explicit conservative
support rule; do not fabricate positive evidence where there are no harmful
training observations. Treat output-form and loss changes as a combined repair
hypothesis unless a matched control isolates them. Bound local runtime with a
real fitting pilot; exact refit/checkpoint/replay requirements remain.

This is a prospective hypothesis, **not yet implemented or trained**. It may fail:
positivity does not guarantee sufficient harm magnitude, locality transfer or
unknown-outcome support. Keep all72 views and all seeds, no threshold/penalty grid,
no future-quality inference, no dropping unknown outcomes, no selected best
locality. Require the same predictive, matched-utility and safety advancement
conditions before any transfer. Independent roles remain closed until justified.

Do not repeat prior global calibration/shrinkage, purely relative-target leaf
refitting or partial-neighbor-only controls. Do not label this cost-head repair
as new neural dynamics or world-model success. The larger goal still requires a
defensible method, independent evidence, strong external comparisons and a full
reproducible paper package.
