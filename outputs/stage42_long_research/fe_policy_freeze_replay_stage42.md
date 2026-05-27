# Stage42-FF FE Policy Freeze / Bootstrap / Replay

- source: `fresh_stage42_fe_policy_freeze_replay`
- generated_at_utc: `2026-05-27T07:12:11.445508+00:00`
- gate: `23 / 23`
- verdict: `stage42_ff_fe_policy_freeze_replay_pass`
- frozen policy hash: `a78db26aa155b38799f5b866f32a2d205018adf2054d9409a016da3163328dff`
- selected candidate: `{'mode': 'fc_to_safety', 'fallback': 'di', 'scope': 'row', 'threshold': 0.05, 'margin': 0.0025}`

## Replayed Test Metrics vs Floor

- all improvement: `26.41%`
- t50 improvement: `23.15%`
- t100 raw-frame diagnostic improvement: `14.01%`
- hard/failure improvement: `24.81%`
- easy degradation: `-31.06%`
- switch rate: `61.95%`
- final near@0.05: `1.32%`

## Exact Replay

- candidate exact replay: `True`
- max metric abs diff vs FE artifact: `0.0`
- max diagnostic abs diff vs FE artifact: `0.0`

## Bootstrap CI

| slice | low | mid | high | n | bootstrap_n |
| --- | ---: | ---: | ---: | ---: | ---: |
| `all` | `26.08%` | `26.40%` | `26.71%` | `47458` | `2000` |
| `t50` | `22.71%` | `23.15%` | `23.55%` | `11538` | `2000` |
| `t100_raw_frame_diagnostic` | `13.46%` | `14.01%` | `14.60%` | `7048` | `2000` |
| `hard_failure` | `24.46%` | `24.81%` | `25.15%` | `35076` | `2000` |
| `easy_degradation` | `-46.77%` | `-45.03%` | `-43.35%` | `11192` | `2000` |

## Near@0.05 Bootstrap CI

| quantity | low | mid | high | n | bootstrap_n |
| --- | ---: | ---: | ---: | ---: | ---: |
| `final_near_005` | `1.22%` | `1.32%` | `1.42%` | `47458` | `2000` |
| `delta_final_minus_fc` | `-0.60%` | `-0.54%` | `-0.47%` | `47458` | `2000` |
| `delta_final_minus_di` | `-0.10%` | `-0.06%` | `-0.02%` | `47458` | `2000` |
| `delta_final_minus_fb` | `0.18%` | `0.23%` | `0.28%` | `47458` | `2000` |

## No Leakage / Claim Boundary

- policy is frozen from validation-selected Stage42-FE candidate.
- replay performs no threshold reselection and no test tuning.
- future waypoints/endpoints are labels only, never inference inputs.
- dataset-local/raw-frame 2.5D only; no metric/seconds claim.
- Stage5C not executed; SMC not enabled.
