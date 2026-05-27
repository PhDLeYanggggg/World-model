# Stage42-FC Objective-Level Proximity Training

- source: `fresh_stage42_objective_level_proximity_training`
- generated_at_utc: `2026-05-27T06:26:41.176525+00:00`
- gate: `22 / 23`
- verdict: `stage42_fc_objective_level_proximity_training_positive_not_promoted`
- selected objective: `label_proximity_objective`
- feature mode: `stage42_am_features`
- lambda: `10.0`

## Protected Test Metrics vs Floor

- all improvement: `26.37%`
- t50 improvement: `23.01%`
- t100 raw-frame diagnostic improvement: `14.02%`
- hard/failure improvement: `24.76%`
- easy degradation: `-31.10%`
- switch rate: `61.99%`

## Ungated Candidate

- all improvement: `47.52%`
- t50 improvement: `42.45%`
- hard/failure improvement: `46.91%`
- easy degradation: `-30.77%`

## Proximity Diagnostics

- protected near@0.05: `1.86%`
- floor near@0.05: `2.24%`
- ungated near@0.05: `2.08%`
- delta near@0.05 vs Stage42-DI: `0.48%`
- delta near@0.05 vs Stage42-FB: `0.76%`

## Comparison To Prior

| reference | all | t50 | t100 raw | hard/failure | easy |
| --- | ---: | ---: | ---: | ---: | ---: |
| `delta_vs_stage42_am` | `1.79%` | `0.99%` | `-0.34%` | `1.01%` | `-5.44%` |
| `delta_vs_stage42_di` | `1.66%` | `0.64%` | `-0.33%` | `0.87%` | `-5.47%` |
| `delta_vs_stage42_fb` | `1.72%` | `0.82%` | `-0.33%` | `0.94%` | `-5.46%` |

## Decision

- promote objective-level training: `False`
- diagnostic positive: `True`
- decision: `objective_level_training_not_enough_keep_stage42_di_or_cq_floor`
- reason: Promotion requires positive all+hard, easy safe, no all/hard loss vs Stage42-DI/FB, and no worse near@0.05 than Stage42-DI.

## No Leakage / Claim Boundary

- future labels are used only for supervised train loss weighting and evaluation labels, not inference features.
- no central velocity, no test endpoint goals, no test threshold tuning.
- dataset-local/raw-frame 2.5D only; no metric/seconds claim.
- Stage5C not executed; SMC not enabled.
