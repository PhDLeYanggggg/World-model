# Selected Harm Is Already Underestimated on the Training Population

English results addendum, 6 October 2026. This extends the
[completed temporal-supervision experiment](../european_temporal_auxiliary_v1/readout/manuscript_addendum.md).
The underlying TRAIN replay is completed real-data inference; the numerical
evidence used in this manuscript assembly is `cached_verified`. No new model,
development prediction or independent confirmation is produced by this document.

## Question and Design

The temporal auxiliary head reduced global signed-score error relative to its
row-mean control, but worsened selected-policy utility and failed the fixed
easy-harm budget. A possible explanation is unseen-scene shift. Another is
that the head already predicts harm poorly on the population its own decision
rule selects. To distinguish these explanations, we replayed all 216 frozen
heads on their original 24 TRAIN packets, retaining three head seeds and all
12 development localities. We changed no optimizer state, score, threshold or
eligibility rule. Unknown future outcomes did not filter inference decisions.

Each arm contributes 72 source/seed views, not 72 independent scenes. The replay
verified feature and checkpoint hashes, preprocessing and repeated predictions.
The decomposition below was checked with 432 scalar sum-identity assertions;
larger logged counts of scalar summands are not independent assertions.

## Risk Definition and Decomposition

Let S be a policy's selected rows with known outcomes, e the frozen TRAIN-defined
easy event, H the positive excess prediction cost over the reference, and R the
reference cost. For this diagnostic define

```
H_E = sum over S of e*H       R_E = sum over S of e*R
H_hat_E = sum over S of predicted easy harm
R_hat_E = sum over S of predicted easy reference cost
q = 0.02

H_E - q*R_E
  = (H_hat_E - q*R_hat_E)
    + (H_E - H_hat_E)
    + q*(R_hat_E - R_E).
```

The terms respectively measure predicted slack, harm underprediction and the
budget-weighted reference error. Negative terms remain negative. This is an
accounting identity on the same selected known rows, not a causal effect or a
new safety theorem. Dividing by actual R_E and multiplying by 100 gives excess
over the fixed budget in percentage points. Zero-reference support would be
undefined, not silently omitted. The diagnostic risk is H_E/R_E: it is neither
net easy-case ADE degradation nor the probability of a physical collision.

## Results

| Auxiliary supervision | Known TRAIN risk violations /72 | Median selected risk (%) | Median predicted/actual selected harm |
|---|---:|---:|---:|
| None |58|5.371021|0.024865|
| Row mean |57|4.567470|0.037706|
| Temporal |52|4.304727|0.039213|

All 216 heads underestimate selected easy-harm mass. For the temporal arm,
the median predicted/actual harm ratio is 0.705898 over all known TRAIN rows
but 0.039213 on its selected rows. These are two medians of per-view ratios,
not a pooled ratio or a percentage of incorrectly classified trajectories.
Selection exposes a much larger conditional error than the global average.

After averaging repeated views within locality and then weighting localities
equally, the temporal arm has the following decomposition:

| Component | Percentage points of actual selected easy-reference cost |
|---|---:|
| Predicted slack | -1.275936 |
| Harm underprediction | +9.197277 |
| Budget-weighted reference error | -0.457331 |
| Realized excess above the 2% budget | +7.464010 |

This equal-locality mean excess is not the median risk in the first table.
Reference cost is overpredicted in only 5/72 temporal views; its average signed
contribution is protective. Underprediction of selected harm dominates this
particular accounting. The 1,828 unknown selected temporal occurrences remain
unassessed for realized harm. They are not assigned zero cost, and known-outcome
violations cannot be explained away by those missing labels.

## Interpretation

Pure unseen-domain shift cannot be the sole explanation: the budget already
fails under TRAIN resubstitution. This does not rule out additional domain shift,
noisy detector labels, feature insufficiency, finite fitting or optimization
effects. Nor does it prove that a particular loss repair will generalize.
These descriptive TRAIN results receive no independent-confirmation claim or
new significance claim. The earlier locality-bootstrap development intervals
must not be recycled as uncertainty for this separate diagnostic.

The evidence favors examining the easy-harm channel before modifying reference
calibration or increasing model size. The next registered comparison replaces
only its quadratic moment loss with a continuous-cost Poisson-style deviance,
retaining the bounded decoder, four other moments, signed-score losses, sampling,
update budget and risk rule. It is a testable repair hypothesis, not a successful
method result. The earlier unsuccessful harm weighting, selected-group loss and
recalibration controls remain part of the record.

## Reproducibility Boundary of the Successor

The real TRAIN pilot completed, but its 100-step own-loss monitors did not
improve. Thirty-six of the successor's 144 full fits were verified before an
execution investigation. One quadratic control failed historical bitwise replay.
A fresh same-node diagnostic reproduced that state exactly with the unmodified
original trainer, both directly and from the original pilot checkpoint; only
the historical artifact differs. Its largest recorded tensor difference is
2.428889274597168e-6. The precise low-level floating-point cause is unresolved.

An explicit pre-readout amendment requires an exact original-implementation
replay for that one identity, while retaining historical exact checks for the
other 71 controls. It does not relax floating tolerances or scientific gates.
The replacement jobs are submitted; the complete successor training grid and
its new development readout have not yet been verified. This execution issue
is not evidence for or against the proposed loss's scientific value.

The manuscript still lacks a successful independently confirmed intervention
method. Image-local detector-silver data, obs8/pred12 and raw stride12 remain
the contract. There is no metric, seconds-level, human-gold, physical-safety,
true-3D or foundation claim. Stage5C execution and SMC remain disabled.

## Traceable Sources

- [TRAIN replay and decomposition](../european_temporal_auxiliary_v1/false_safe_train_diagnostic_v1/report.md).
- [Frozen aggregate values](../european_temporal_auxiliary_v1/false_safe_train_diagnostic_v1/summary.json).
- [Selected versus all-known harm ratios](../european_temporal_auxiliary_v1/false_safe_train_diagnostic_v1/harm_ratio_audit.json).
- [Earlier unsuccessful controls](../european_temporal_auxiliary_v1/false_safe_train_diagnostic_v1/prior_controls.md).
- [Successor protocol](../european_easy_harm_deviance_v1/protocol.md) and
  [explicit replay amendment](../european_easy_harm_deviance_v1/control_replay_amendment.md).
- [Machine-readable evidence passport](train_selection_evidence.json).
