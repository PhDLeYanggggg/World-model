# Stage42-GE Conversion Capability -> Intake Bridge

- source: `fresh_stage42_ge_conversion_capability_intake_bridge`
- generated_at_utc: `2026-05-27T11:51:20.911043+00:00`
- git_commit: `ac8a2e0`
- input_hash: `c8d15ee4eed1ae931c912166147861d307465e1b34b86d149e9570a3d866f2a6`
- gate: `20 / 20`
- verdict: `stage42_ge_conversion_capability_intake_bridge_pass`

## Role

- This bridges DW source-specific conversion dry-run evidence into the intake as `conversion_capability_prefill`.
- It records source IDs, horizon support, source-CV feasibility, and technical readiness after terms confirmation.
- It does not grant permission, convert data, train, evaluate, or make metric/seconds claims.

## Summary

- intake_rows: `5`
- rows_with_source_specific_dry_run: `2`
- rows_with_source_cv_feasible_after_terms: `1`
- technical_ready_after_terms_sources: `5`
- t50/t100 windows after terms: `10060` / `5696`
- conversion_ready_now: `0`

## Intake Rows

| dataset | sources | tech-ready sources | source-CV after terms | t50 after terms | t100 after terms | conversion ready now |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ucy_crowd_original` | 4 | 3 | True | 9554 | 5605 | False |
| `eth_biwi_original` | 2 | 2 | False | 506 | 91 | False |
| `aerialmpt_or_other_topdown` | 0 | 0 | False | 0 | 0 | False |
| `opentraj_toolkit` | 0 | 0 | False | 0 | 0 | False |
| `trajnetplusplus_official` | 0 | 0 | False | 0 | 0 | False |

## Claim Boundary

- Conversion capability is not legal permission and not conversion readiness.
- UCY has a source-CV-capable plan after terms; ETH has technical source-specific candidates but fewer sources for source-CV.
- Current M3W remains protected dataset-local/raw-frame 2.5D; no true 3D, foundation, metric/seconds, Stage5C, or SMC claim.
