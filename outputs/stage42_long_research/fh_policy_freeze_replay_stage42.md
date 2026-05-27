# Stage42-FI FH Policy Freeze / Bootstrap / Replay

- source: `fresh_stage42_fh_policy_freeze_replay`
- generated_at_utc: `2026-05-27T07:43:44.782843+00:00`
- gate: `25 / 25`
- verdict: `stage42_fi_fh_policy_freeze_replay_pass`
- frozen policy hash: `f1f6e0636167fae8721a3f7195f188dcbe1a83194b04fa0625b378ad38b5aed6`
- selected candidate: `{'mode': 'fc_to_safety', 'fallback': 'di', 'scope': 'row', 'threshold': 0.05, 'margin': 0.01}`
- internal_val_group: `UCY::UCY/zara03/crowds_zara03.txt`

## Replayed Test Metrics vs Floor

- all improvement: `34.98%`
- t50 improvement: `28.97%`
- t100 raw-frame diagnostic improvement: `20.57%`
- hard/failure improvement: `33.10%`
- easy degradation: `-36.91%`
- switch rate: `74.64%`
- final near@0.05: `1.26%`

## By Domain

| domain | rows | all | t50 | t100 raw | hard/failure | easy | positive_safe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `TrajNet` | 37918 | 34.21% | 29.35% | 18.47% | 32.36% | -36.75% | True |
| `UCY` | 9540 | 37.49% | 27.63% | 26.99% | 35.43% | -37.79% | True |

## Exact Replay

- candidate exact replay: `True`
- max metric abs diff vs FH artifact: `0.0`
- max diagnostic abs diff vs FH artifact: `0.0`

## Bootstrap CI

| slice | low | mid | high | n | bootstrap_n |
| --- | ---: | ---: | ---: | ---: | ---: |
| `all` | `34.62%` | `34.98%` | `35.32%` | `47458` | `2000` |
| `t50` | `28.46%` | `28.97%` | `29.48%` | `11538` | `2000` |
| `t100_raw_frame_diagnostic` | `19.96%` | `20.57%` | `21.27%` | `7048` | `2000` |
| `hard_failure` | `32.73%` | `33.09%` | `33.48%` | `35076` | `2000` |
| `easy_degradation` | `-60.74%` | `-58.49%` | `-56.32%` | `11192` | `2000` |

## Near@0.05 Bootstrap CI

| quantity | low | mid | high | n | bootstrap_n |
| --- | ---: | ---: | ---: | ---: | ---: |
| `final_near_005` | `1.15%` | `1.25%` | `1.35%` | `47458` | `2000` |
| `delta_final_minus_fc` | `-0.79%` | `-0.71%` | `-0.64%` | `47458` | `2000` |
| `delta_final_minus_di` | `-0.10%` | `-0.06%` | `-0.02%` | `47458` | `2000` |
| `delta_final_minus_fb` | `0.21%` | `0.27%` | `0.32%` | `47458` | `2000` |

## No Leakage / Claim Boundary

- policy is frozen from validation-selected Stage42-FH candidate.
- replay performs no threshold reselection and no test tuning.
- UCY internal validation is from original train sources only.
- future waypoints/endpoints are labels only, never inference inputs.
- dataset-local/raw-frame 2.5D only; no metric/seconds claim.
- Stage5C not executed; SMC not enabled.
