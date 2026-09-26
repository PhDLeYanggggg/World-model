# Nested Residual Provenance: Source-Code Preflight

Status: fresh source inspection, not a fitted OOF experiment or passed gate.
The completed context protocol, estimators and predictions are unchanged.

## Verified Dependencies

- `scripts/run_m3w_european_selected_risk_learning.py:63` constructs matched
  producer/controller roles. The group's producer roster is disjoint from
  controller B; reuse must remain group-specific, not borrow another group's
  predictor that may have seen the current outer-held locality.
- `scripts/run_m3w_european_task_gradients.py:52` removes the outer locality
  before retrieving fitting future labels and computing cost supervision.
- `src/world_model/m3w_native_gain_harm.py:40` computes feature mean/std,
  cost scale, equal-locality weights and the positive-CV-error 25th-percentile
  easy cut from its supplied fitting rows. It supports two fitting localities,
  but that is interface feasibility, not demonstrated adequate support.
- `src/evaluation/m3w_harm_tail_diagnostics.py:17` defines the easy-cost
  target from that fitting-only cut. The cut is label-derived, not a harmless
  input constant that can be inherited from rows being treated as inner-held.
- `src/world_model/m3w_membership_auxiliary.py:36` additionally learns
  initialization and loss normalization from its fitting labels. These must
  also be rebuilt within the inner fold. Checkpoint resume is identity-bound.

## Required Next-Experiment Constraints

For each outer-held view, the remaining three controller localities permit
three inner folds: fit on two, predict on the third. Rebuild all learned
preprocessing, initialization, scales, easy cut and label definitions from
those two fitting localities. Evaluate each OOF residual using its own inner
model's label rule; do not apply a cut estimated using the residual labels.
Retain row IDs, outer/inner rosters, cut values, producer hashes and complete
normalization provenance. Never substitute an existing outer-fold checkpoint
that trained on the current outer-held locality.

The final outer readout must retain the existing outer fitting-only target
definition and comparator. Inner cuts can differ from that outer cut, so a
pooled OOF residual probe also faces target-definition and training-size
transport. Report those differences and design matched controls before
calling a result a pure residual-provenance effect. Changing the primary easy
definition to a producer-derived cut would be a different experiment, not an
unannounced implementation fix.

For the original cost estimator alone, 144 views times three inner fits
would require 432 new cost heads, plus the fixed residual probes. This is a
budget count, not a training receipt or an approved registration. A pilot,
fitting-only support check and explicit frozen configuration must precede
that work. Keep the seven features and strong controls; do not select the
closing-speed feature from a favorable development-held sign count.

No OOF heads have been trained in this record. No independent role, target
rule, threshold, policy or deployment has been changed. Stage5C and SMC remain
off. Detector pixels and annotation steps remain unverified for metric/time
claims; the current result is not a foundation or true3D world model.
