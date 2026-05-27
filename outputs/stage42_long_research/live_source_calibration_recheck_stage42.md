# Stage42-GA Live Source / Calibration Recheck

- source: `fresh_stage42_live_source_calibration_recheck`
- generated_at_utc: `2026-05-27T11:17:24.110667+00:00`
- git_commit: `c335920`
- input_hash: `f74dc2720e316a3a706624da5343d7cd7cdc9b10226211171fcc5ee859ac1afe`
- gate: `15 / 15`
- verdict: `stage42_ga_live_source_calibration_recheck_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- GA fresh-scans local paths and cached legal/calibration state; it does not download, convert, train, or evaluate.
- Local files are not counted as legal conversion readiness without explicit terms/path/source identity confirmation.
- Stage5C latent generative is not executed; SMC is not enabled.

## Summary

- targets_audited: `7`
- local_path_found_targets: `7`
- existing_converted_artifact_targets: `1`
- new_conversion_ready_targets: `0`
- source_action_conversion_ready_now: `0`
- unified_queue_count: `0`
- highest_priority_next_action: `FW-TERMS-ucy_crowd_original`

## Target Rows

| target | domain | local path | converted/cached | new ready | calibration | blockers | next action |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| `sdd` | `SDD` | True | True | False | `pixel_raw_frame_only` | none | Do not relabel as metric/seconds-level; next useful action is homography/scale/FPS verification only. |
| `opentraj_toolkit` | `OpenTraj` | True | False | False | `not_verified_this_target` | explicit_terms_or_source_identity_not_confirmed | Confirm official terms and fill `opentraj_toolkit` fields in source_terms_confirmation_template_stage42.json: terms_accepted_by_user, terms_acceptance_date, allowed_use, local_path, source_identity. |
| `eth_biwi_original` | `ETH_UCY` | True | False | False | `not_verified_this_target` | explicit_terms_or_source_identity_not_confirmed | Confirm official terms and fill `eth_biwi_original` fields in source_terms_confirmation_template_stage42.json: terms_accepted_by_user, terms_acceptance_date, allowed_use, local_path, source_identity. |
| `trajnetplusplus_official` | `TrajNet` | True | False | False | `not_verified_this_target` | explicit_terms_or_source_identity_not_confirmed<br>h100_long_source_support_not_closed | Confirm official terms and fill `trajnetplusplus_official` fields in source_terms_confirmation_template_stage42.json: terms_accepted_by_user, terms_acceptance_date, allowed_use, local_path, source_identity. |
| `ucy_crowd_original` | `UCY` | True | False | False | `not_verified_this_target` | explicit_terms_or_source_identity_not_confirmed<br>ucy_terms_and_h100_candidate_confirmation_missing | Confirm official terms and fill `ucy_crowd_original` fields in source_terms_confirmation_template_stage42.json: terms_accepted_by_user, terms_acceptance_date, allowed_use, local_path, source_identity. |
| `tgsim` | `TGSIM` | True | False | False | `traffic_metric_diagnostic_only` | explicit_terms_or_source_identity_not_confirmed<br>diagnostic_only_not_topdown_pedestrian_claim | Keep as traffic diagnostic-only evidence; do not convert or report it as pedestrian/top-down world-model success. |
| `aerialmpt_or_other_topdown` | `AerialMPT` | True | False | False | `not_verified_this_target` | explicit_terms_or_source_identity_not_confirmed<br>official_url_or_terms_not_verified | Confirm official terms and fill `aerialmpt_or_other_topdown` fields in source_terms_confirmation_template_stage42.json: terms_accepted_by_user, terms_acceptance_date, allowed_use, local_path, source_identity. |

## Gate

| gate | pass |
| --- | ---: |
| `source_fresh` | True |
| `all_required_targets_audited` | True |
| `local_scan_performed` | True |
| `data_calibration_loaded` | True |
| `source_action_loaded` | True |
| `unified_queue_loaded` | True |
| `no_new_conversion_ready_overclaim` | True |
| `source_action_ready_zero_preserved` | True |
| `every_blocked_target_has_next_action` | True |
| `user_action_written` | True |
| `no_download_conversion_eval` | True |
| `no_metric_seconds_overclaim` | True |
| `no_true3d_foundation_overclaim` | True |
| `stage5c_false` | True |
| `smc_false` | True |

## Interpretation

- 本机有多个 external/raw/cache artifact，但本轮没有任何新 source 达到 guarded conversion readiness。
- SDD 仍是 pixel/raw-frame；external 仍是 dataset-local/raw-frame 或 diagnostic。
- 下一步最高优先级仍是填 UCY official terms/path/source identity，然后 rerun terms validator、guarded queue、conversion/no-leakage/source-CV。
