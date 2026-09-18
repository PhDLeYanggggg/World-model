# Cost Deferral Analysis Contract

This consumer is frozen during the six-branch training run, after observing
the first expected-cost branch's training milestones, before joint analysis.
It is not presented as a pre-fit analysis registration. Training objectives,
budget, seeds, score threshold and data restrictions remain the pre-fit ones
in `configs/m3w_source_cost_deferral_v1.json`, committed at `e91a7a37`.

## Fixed Checks

- Six completed continuations, 48,000 additional updates, 24 exact milestone
  prediction/score replays; three verified cached dense controls.
- Identical sampled counts and sampler/Torch states for all same-seed arms at
  each milestone, plus explicit reconstruction of the sampled streams.
- Parent proposals reproduce exactly at step 2,000. New score is zero; emitted
  action is the exact stationary baseline. The causal action threshold stays 0.
- Bounded finite proposals, zero unsupported output and exact zero fallback.
- Recompute stored training metrics from aligned predictions and labels.
- Read-only completed resume with zero new updates and artifact hashes stable.

## Training Diagnostics Only

Show every seed and milestone. Contrast both raw proposals and emitted hard
actions with stationary CV and the same-seed dense control. Report moving and
hard slices, absolute zero-target harm, intervention rate, tails and per-site
training results. Three-seed ranges are optimization variability, not held-data
confidence intervals. The easy percentage is undefined when baseline error is
zero; an absolute-harm comparison does not change the approved main easy gate.

Expected soft-action cost is not the accuracy of an executed deterministic
forecast. A sigmoid is not a calibrated probability. Signed-cost fitting is
only interpretable as cost fitting for the supervised-cost variant, and even
then is in-sample fitting against jointly changing candidate errors. Include
constant-mean and zero-score comparisons; do not call this risk calibration.

The pre-fit training signal requires positive gain over CV and matched dense
control, positive moving or hard gain, and no greater absolute easy harm than
the dense control. All-baseline/zero-gain outputs fail to establish predictive
success. No seed or milestone is selected as a deployable winner.

The consumer cannot forecast held bookstore/main roles. They remain unscored
in this experiment. The generalization, scene-level joint decision and
independent calibration goals remain open regardless of these training scores.
Source data retain offline supplied-annotation, pixel/past-normalized and
raw-frame status. No Stage5C execution, SMC, metric or seconds claims.
