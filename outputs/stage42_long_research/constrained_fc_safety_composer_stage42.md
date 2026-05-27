# Stage42-FE Constrained FC/Safety Composer

- source: `fresh_stage42_constrained_fc_safety_composer`
- generated_at_utc: `2026-05-27T07:03:24.897323+00:00`
- gate: `19 / 19`
- verdict: `stage42_fe_constrained_fc_safety_composer_pass_promotable`
- decision: `promote_stage42_fe_constrained_fc_safety_composer`
- selected candidate: `{'mode': 'fc_to_safety', 'fallback': 'di', 'scope': 'row', 'threshold': 0.05, 'margin': 0.0025}`

## Protected Test Metrics vs Floor

- all improvement: `26.41%`
- t50 improvement: `23.15%`
- t100 raw-frame diagnostic improvement: `14.01%`
- hard/failure improvement: `24.81%`
- easy degradation: `-31.06%`
- switch rate: `61.95%`

## References

| policy | all | t50 | t100 raw | hard/failure | easy | near@0.05 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fc_test` | `26.37%` | `23.01%` | `14.02%` | `24.76%` | `-31.10%` | `1.86%` |
| `di_test` | `24.72%` | `22.36%` | `14.35%` | `23.89%` | `-25.63%` | `1.38%` |
| `fb_test` | `24.65%` | `22.19%` | `14.35%` | `23.82%` | `-25.64%` | `1.10%` |
| `fa_test` | `24.61%` | `22.05%` | `14.36%` | `23.77%` | `-25.67%` | `1.21%` |

## Diagnostics

- final near@0.05: `1.32%`
- delta near@0.05 vs FC: `-0.54%`
- delta near@0.05 vs DI: `-0.06%`
- delta near@0.05 vs FB: `0.23%`
- fallback rate: `1.66%`

## Delta vs Prior

- delta vs FC: `{'all_improvement': 0.0003715095874463614, 't50_improvement': 0.001446843202881798, 't100_raw_frame_diagnostic_improvement': -0.0001084341485075857, 'hard_failure_improvement': 0.00047883004420823383, 'easy_degradation': 0.0004427747622699485}`
- delta vs DI: `{'all_improvement': 0.01692732018015164, 't50_improvement': 0.007878549453218087, 't100_raw_frame_diagnostic_improvement': -0.0033631779829500497, 'hard_failure_improvement': 0.00920100213144004, 'easy_degradation': -0.0542880221359916}`
- delta vs FB: `{'all_improvement': 0.017614022896018322, 't50_improvement': 0.009653702988816626, 't100_raw_frame_diagnostic_improvement': -0.0033979814877990178, 'hard_failure_improvement': 0.009921735700743062, 'easy_degradation': -0.05419582041366533}`

## No Leakage / Claim Boundary

- composer uses predicted rollout geometry only, not future labels.
- future labels are evaluation labels only.
- no central velocity, no test endpoint goals, no test threshold tuning.
- dataset-local/raw-frame 2.5D only; no metric/seconds claim.
- Stage5C not executed; SMC not enabled.
