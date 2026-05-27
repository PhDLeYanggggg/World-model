# Stage42-FH UCY-Supported FE Composer

- source: `fresh_stage42_ucy_supported_fe_composer`
- generated_at_utc: `2026-05-27T07:34:35.615648+00:00`
- gate: `20 / 20`
- verdict: `stage42_fh_ucy_supported_fe_composer_pass`
- decision: `promote_stage42_fh_ucy_supported_fe_composer`
- selected candidate: `{'mode': 'fc_to_safety', 'fallback': 'di', 'scope': 'row', 'threshold': 0.05, 'margin': 0.01}`

## Why This Exists

- Stage42-FG showed frozen FE is robust on TrajNet but fallback-only on UCY.
- Stage42-FH repairs the validation support rather than changing test thresholds: UCY internal validation is carved from original train sources only.

## Protected Test Metrics vs Floor

- all improvement: `34.98%`
- t50 improvement: `28.97%`
- t100 raw-frame diagnostic improvement: `20.57%`
- hard/failure improvement: `33.10%`
- easy degradation: `-36.91%`
- switch rate: `74.64%`
- final near@0.05: `1.26%`

## By Domain

| domain | rows | all | t50 | t100 raw | hard/failure | easy | switch | positive_safe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `TrajNet` | 37918 | 34.21% | 29.35% | 18.47% | 32.36% | -36.75% | 77.54% | True |
| `UCY` | 9540 | 37.49% | 27.63% | 26.99% | 35.43% | -37.79% | 63.13% | True |

## Bootstrap CI

| slice | low | mid | high | n | bootstrap_n |
| --- | ---: | ---: | ---: | ---: | ---: |
| `all` | `34.64%` | `34.98%` | `35.33%` | `47458` | `1000` |
| `t50` | `28.47%` | `28.97%` | `29.46%` | `11538` | `1000` |
| `t100_raw_frame_diagnostic` | `19.85%` | `20.57%` | `21.23%` | `7048` | `1000` |
| `hard_failure` | `32.70%` | `33.08%` | `33.50%` | `35076` | `1000` |
| `easy_degradation` | `-60.85%` | `-58.52%` | `-56.49%` | `11192` | `1000` |

## No Leakage / Claim Boundary

- UCY internal validation is selected from original train sources only.
- test rows are unchanged and used once for final evaluation.
- no future endpoint/waypoint input, no central velocity, no test endpoint goals, no test threshold tuning.
- dataset-local/raw-frame 2.5D only; no metric/seconds claim.
- Stage5C not executed; SMC not enabled.
