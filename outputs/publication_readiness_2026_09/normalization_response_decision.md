# Observed-Unit Frame and Gradient Response Diagnostic

This follows completed source-mechanism controls, not a new broad model sweep.
Hypothesis: fixed native-unit floors and mixed raw/frame-based summaries create
stationary source/main conditioning differences. Analytic log-loss derivatives
do not establish actual model-gradient contributions; measure those first.

Keep the approved main primary, 11,966-window fit cohort, original SDD train40,
and closed development/calibration/confirmation roles unchanged. Use only main
fold-0 training rows for gradient probes. Cached held arrays may be read for file
integrity as before, but no held targets or metrics enter this diagnostic.
Source event labels are training-only supervised strata, never inference inputs.

Implement a new, separate internal representation over the bound 476-column
past feature schema. Normalize positions by the maximum observed ego/selected
neighbor radius, re-express in a past-only direction, and reconstruct motion
summaries with horizon-relative time. Do not retain raw length, speed,
acceleration, raw horizon or raw stride shortcuts. Preserve partial neighbor
masks. If the complete observed spatial context is zero, no positive length is
identifiable; its correction must be zero rather than inventing a scale.
Restore learned deltas to the unchanged evaluation coordinates before scoring.
This repairs a different typed schema than the older Transformer conditioner;
unit tests alone do not establish a forecasting gain.

Synthetic tests: coordinate rescalings 1e-4/1e4, stationary and tiny moving
histories, translation, future mutation, absent spatial support, feature-mask
validation and output restoration. No labels in the input transform.

Gradient probe: seeds17/29/43, geometry/mask/RGB, initial legacy model, frozen
real-source final legacy model, initial unit frame with restored primary log
loss, and initial unit frame with internal-coordinate log loss. Select at most
32 fixed complete-label supported windows per event/domain, before probing;
retain event population sizes and sampled track counts. Measure individual full
parameter-gradient norms, head/encoder parts, cancellation and population-weighted
estimates. No optimizer updates, no held-score selection and no deployment.
Small correlated window probes are diagnostic, not independent statistics.

The actual source checkpoint was trained on source and main fit labels; it is
used to inspect its training response, not to claim out-of-sample performance.
The alternative internal loss is tested as an optimization hypothesis only.
Any subsequent fitting requires a separate fixed comparison; no automatic
promotion follows from prettier gradient norms or invariant features.

Source annotations are offline/silver; no strict sensor-as-of claim. No metric,
seconds-level, true-3D, foundation, Stage5C or SMC claim. The long-term scientific
goal is not achieved by this diagnostic.
