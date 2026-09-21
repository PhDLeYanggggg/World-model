# Frozen Cost-Head Fit Diagnosis

2026-09-20, post-hoc development diagnosis. The preceding risk study found both
observed underprediction and a global-versus-easy constraint mismatch. A first
hash-checked read of six existing ridge heads and their OOF training arrays
already suggests that low predicted risk is unreliable within cost-head training
rows, before development-domain shift. This design is not blinded registration.

Retain both predictor families, seeds17/29/43 and both ridge/neural cost heads.
Use complete hash-bound fit-only OOF caches and fixed final checkpoints. The
trajectory producer excludes its target fold; the cost head, however, was fitted
on all these OOF rows. These diagnostics are in-sample for that head, not held-out
cost calibration or generalization. Do not call them test results.

Check targets, benefit/harm ordering, normalization, parent identities, saved
coefficients and nonnegative output handling. Quantify full-fit cost MSE against
the fitted-label mean, negative raw ridge predictions clipped to zero, true costs
among those clipped rows, and conditional cost errors among rows meeting the
original fixed per-agent conservative/moderate eligibility thresholds. These are
eligibility sets, not scene-solver selections or actual intervention rates.
Report both policies and every seed, including empty sets.

Also describe each fit recording and the share of squared target mass in the
largest 1% of harm labels. The latter is a fixed descriptive slice, not a new
training weight or clipping threshold. Fit labels never become inference inputs.
No new neural forecasts, new training, primary metric, policy, threshold, future
mask or data-role change. Do not expose new development/main/test/confirmation
labels. Replay existing neural cost heads using their original family device;
no CPU/NumPy fallback disguised as MPS inference.

This can distinguish a fit/readout issue already present on training rows from
a problem appearing only after cross-site transfer. It cannot by itself isolate
optimization, model misspecification and feature insufficiency, or establish that
another head would solve easy preservation. Clipping to zero never decreases a
negative estimate, so do not claim that removing clipping is a safety fix.
Keep frozen experiment-bound code and metrics unchanged. The primary-metric
decision remains pending. Stage5C/SMC disabled; no deployment or submission claim.
