# Stage42-GC Prefill -> Intake Bridge

- source: `fresh_stage42_gc_prefill_intake_bridge`
- generated_at_utc: `2026-05-27T11:30:48.711818+00:00`
- git_commit: `77ea60b`
- input_hash: `7ca90654a0328643b2554fccec90f14f877a5af5a852bd7184d3ed1a9ad49dc3`
- gate: `16 / 16`
- verdict: `stage42_gc_prefill_intake_bridge_pass`

## Role

- This bridges GB path/source-identity suggestions into the EH intake template as `prefill_suggestion` hints.
- It does not fill `user_confirmation`, accept terms, download, convert, train, evaluate, or mark any dataset ready.
- The validator must still block every row until the user manually confirms terms/path/source identity.

## Summary

- intake_rows: `5`
- rows_with_prefill_suggestion: `5`
- rows_with_user_confirmation: `0`
- conversion_ready_now: `0`
- snapshot: `outputs/stage42_long_research/source_terms_confirmation_intake_prefilled_snapshot_stage42.json`
- updated intake template: `outputs/stage42_long_research/source_terms_confirmation_intake_template_stage42.json`

## Intake Rows

| dataset | suggested local path | user confirmation filled | conversion ready |
| --- | --- | ---: | ---: |
| `ucy_crowd_original` | `external_data/OpenTraj/datasets/UCY` | False | False |
| `eth_biwi_original` | `external_data/OpenTraj/datasets/ETH` | False | False |
| `aerialmpt_or_other_topdown` | `data/aerialmpt/DLR_AerialMPT_Dataset.zip` | False | False |
| `opentraj_toolkit` | `external_data/OpenTraj` | False | False |
| `trajnetplusplus_official` | `external_data/OpenTraj/datasets/TrajNet` | False | False |

## Claim Boundary

- Prefill suggestions are not legal permission and not source conversion readiness.
- Current M3W remains protected dataset-local/raw-frame 2.5D; no true 3D, foundation, metric, seconds-level, Stage5C, or SMC claim.
