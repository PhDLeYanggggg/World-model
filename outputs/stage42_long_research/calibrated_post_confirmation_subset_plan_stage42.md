# Stage42-GH Calibrated Post-Confirmation Subset Plan

- source: `fresh_stage42_gh_calibrated_post_confirmation_subset_plan`
- generated_at_utc: `2026-05-27T12:15:57.831470+00:00`
- git_commit: `72f6f05`
- input_hash: `eb75dd9831120398b299aed3264fb08502d96c027877d737f5bbadc5da467979`
- gate: `14 / 14`
- verdict: `stage42_gh_calibrated_post_confirmation_subset_plan_pass`

## Role

- Combines GF source-level conversion planning with BN source-level time/geometry evidence.
- Identifies restricted metric/time subset candidates after user-confirmed terms/path/source identity.
- Does not download, convert, evaluate, or allow global metric/seconds claims.

## Summary

- planned_source_rows: `6`
- restricted_metric_time_candidates_after_terms: `5`
- restricted_ready_now: `0`
- calibrated t50/t100 windows after terms: `10060` / `5696`
- domains_with_candidates: `ETH_UCY, UCY`

## Dataset Summary

| dataset | domain | sources | calibrated candidates | calibrated t50 | calibrated t100 | top calibrated source |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `ucy_crowd_original` | `UCY` | 4 | 3 | 9554 | 5605 | `UCY_students03` |
| `eth_biwi_original` | `ETH_UCY` | 2 | 2 | 506 | 91 | `ETH_seq_eth` |

## Top Calibrated Candidate Rows

| rank | dataset | source | local claim after legal conversion | t50 | t100 | h50 seconds hint | ready now |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | `ucy_crowd_original` | `UCY_students03` | `source_specific_annotation_step_meter_coordinate_evidence` | 6491 | 3413 | 20.0 | False |
| 2 | `ucy_crowd_original` | `UCY_zara02` | `source_specific_annotation_step_meter_coordinate_evidence` | 2823 | 2095 | 20.0 | False |
| 3 | `ucy_crowd_original` | `UCY_zara01` | `source_specific_annotation_step_meter_coordinate_evidence` | 240 | 97 | 20.0 | False |
| 4 | `eth_biwi_original` | `ETH_seq_eth` | `source_specific_annotation_step_meter_coordinate_evidence` | 291 | 91 | 20.0 | False |
| 5 | `eth_biwi_original` | `ETH_seq_hotel` | `source_specific_annotation_step_meter_coordinate_evidence` | 215 | 0 | 20.0 | False |

## Claim Boundary

- This plan is a post-confirmation candidate map, not permission and not converted data.
- `restricted_ready_now` remains zero because user-confirmed terms/path/source identity is still absent.
- Source-level H/FPS evidence may support a future restricted subset only after guarded conversion and no-leakage evaluation.
- Global M3W remains dataset-local/raw-frame 2.5D; no true-3D, foundation, global metric, seconds-level, Stage5C, or SMC claim is allowed.
