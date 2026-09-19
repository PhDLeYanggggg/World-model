# Single-factor readout conditioning comparison

The fixed-checkpoint training diagnostic found full-gradient norms of 80.04 to
117.54, with at least 99.99839% of squared gradient energy in the final output
layer. All 6,144 sampled batch gradients exceed cap 5, but their mean clipped
directions remain closely aligned with the full population gradient. This is
not evidence that clipping reverses the optimization direction. It motivates
testing the output parameterization, not simply removing safety clipping.

## Fixed Intervention

Retain the preceding 24-head matrix: geometry/centered appearance, the same four
explored source folds, seeds 17/29/43, 10,000 updates per head, importance-corrected
uniform-row ADE, identical sample streams, AdamW, learning-rate schedule and
cap-5 clipping. Initialize again from the same seed, not a selected checkpoint.

Replace the pre-bound raw readout q with q/c, where
c = max(1, median(training restoration_radius / existing training loss_scale)).
The median is over all training-complement rows, including unsupported radii.
No held statistic contributes. The existing loss scale was already estimated
from training labels; it is not a past-only feature and is never presented as
one. The full per-row bound and output support remain unchanged. Multiplying
the final affine layer's weights and bias by c recovers every old forecast, so
the represented function class is unchanged. Optimizer behavior, effective
parameter step sizes and weight-decay geometry can change; do not call this
a pure gradient-clipping ablation or a proven information repair.

## Evaluation and Stopping

Complete the fixed budget, save atomic checkpoints every 200 updates and replay
every model. Compare all 24 endpoints with their matched unconditioned controls.
Use the unchanged conditional four-site, three-seed analysis and 2,000 resamples.
No best-seed, early-stop, alpha, threshold or checkpoint selection from held
results. No new main/outer/bookstore/external evaluation. The source folds have
already been explored and cannot become independent confirmation by renaming.

Primary interpretation: does numerical conditioning yield useful candidate
forecasts, rather than merely removing tiny static jitter? Report whole-cohort,
nonzero-target and hard-slice gain, static absolute pixel harm and binary-oracle
headroom. Zero-error static CV makes percentage easy degradation undefined.
Negative results remain negative. This experiment does not certify deployment,
strict real-time observation, metric/seconds/3D/foundation capability, Stage5C
or SMC. The main 8/12 protocol and its risk rule are not changed.
