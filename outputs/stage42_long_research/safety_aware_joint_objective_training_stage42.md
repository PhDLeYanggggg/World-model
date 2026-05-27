# Stage42-FD Safety-Aware Joint Objective Training

- source: `fresh_stage42_safety_aware_joint_objective_training`
- generated_at_utc: `2026-05-27T06:50:14.534349+00:00`
- gate: `22 / 26`
- verdict: `stage42_fd_safety_aware_joint_objective_positive_not_promoted`
- selected objective: `fc_label_proximity_control`
- feature mode: `stage42_am_features`
- lambda: `100.0`
- teacher alpha: `0.0`
- safety mask: `none`

## Protected Test Metrics vs Floor

- all improvement: `26.33%`
- t50 improvement: `22.70%`
- t100 raw-frame diagnostic improvement: `14.02%`
- hard/failure improvement: `24.69%`
- easy degradation: `-31.11%`
- switch rate: `61.99%`

## Ungated Candidate

- all improvement: `47.39%`
- t50 improvement: `41.85%`
- hard/failure improvement: `46.74%`
- easy degradation: `-30.89%`

## Proximity Diagnostics

- protected near@0.05: `1.86%`
- floor near@0.05: `2.24%`
- ungated near@0.05: `2.08%`
- delta near@0.05 vs Stage42-DI: `0.48%`
- delta near@0.05 vs Stage42-FB: `0.77%`
- delta near@0.05 vs Stage42-FC: `0.01%`

## Comparison To Prior

| reference | all | t50 | t100 raw | hard/failure | easy |
| --- | ---: | ---: | ---: | ---: | ---: |
| `delta_vs_stage42_am` | `1.76%` | `0.68%` | `-0.35%` | `0.94%` | `-5.45%` |
| `delta_vs_stage42_di` | `1.62%` | `0.34%` | `-0.33%` | `0.80%` | `-5.48%` |
| `delta_vs_stage42_fa` | `1.73%` | `0.65%` | `-0.35%` | `0.92%` | `-5.45%` |
| `delta_vs_stage42_fb` | `1.69%` | `0.52%` | `-0.33%` | `0.87%` | `-5.47%` |
| `delta_vs_stage42_fc` | `-0.04%` | `-0.31%` | `-0.00%` | `-0.07%` | `-0.01%` |

## Decision

- promote safety-aware objective: `False`
- diagnostic positive: `True`
- decision: `safety_aware_objective_not_enough_keep_stage42_di_or_cq_floor`
- reason: Promotion requires positive all+hard, easy safe, no all/hard loss vs Stage42-DI/FB/FC, and no worse near@0.05 than Stage42-DI.

## No Leakage / Claim Boundary

- FA teacher is used only as train loss regularizer; it is not an inference feature.
- future labels are used only for supervised train loss and evaluation labels.
- no central velocity, no test endpoint goals, no test threshold tuning.
- dataset-local/raw-frame 2.5D only; no metric/seconds claim.
- Stage5C not executed; SMC not enabled.
