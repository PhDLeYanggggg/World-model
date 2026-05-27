# Stage42-GD Calibration Hint -> Intake Bridge

- source: `fresh_stage42_gd_calibration_hint_intake_bridge`
- generated_at_utc: `2026-05-27T11:36:46.203940+00:00`
- git_commit: `0283496`
- input_hash: `09c2f2002b1ee674cdb30be01de736e36a18195aad1f5f49a412e87261669120`
- gate: `18 / 18`
- verdict: `stage42_gd_calibration_hint_intake_bridge_pass`

## Role

- This bridges DU metadata-only H/FPS/stride hints into the source terms intake as `calibration_prefill`.
- It does not accept terms, convert, train, evaluate, or create metric/seconds claims.
- Metric/time hints remain source-specific leads only until legal confirmation and calibration validation pass.

## Summary

- intake_rows: `5`
- rows_with_calibration_prefill: `5`
- rows_with_any_calibration_hint: `3`
- rows_with_metric_time_subset_hint: `2`
- conversion_ready_now: `0`
- metric/seconds claim allowed now: `False` / `False`

## Intake Rows

| dataset | H hints | time hints | stride hints | metric/time subset hint | claim allowed |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ucy_crowd_original` | 7 | 1 | 9 | True | False |
| `eth_biwi_original` | 2 | 2 | 4 | True | False |
| `aerialmpt_or_other_topdown` | 0 | 0 | 0 | False | False |
| `opentraj_toolkit` | 0 | 0 | 0 | False | False |
| `trajnetplusplus_official` | 0 | 1 | 97 | False | False |

## Claim Boundary

- Calibration hints are not source conversion readiness.
- Calibration hints are not global metric or seconds-level evidence.
- Current M3W remains protected dataset-local/raw-frame 2.5D; no true 3D, foundation, Stage5C, or SMC claim.
