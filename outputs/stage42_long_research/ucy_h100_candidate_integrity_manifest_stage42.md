# Stage42-GX UCY H100 Candidate Integrity Manifest

- source: `fresh_stage42_gx_ucy_h100_candidate_integrity_manifest`
- generated_at_utc: `2026-05-27T15:08:36.390226+00:00`
- git_commit: `0eafd4b`
- gate: `17 / 17`
- verdict: `stage42_gx_ucy_h100_candidate_integrity_manifest_pass`
- result source: `fresh_run file integrity manifest from cached_verified blocker decisions`

## Summary

- candidate_rows: `6`
- existing_files: `6`
- target_family_candidates: `2`
- t100_capable_files: `6`
- total_parsed_rows: `98032`
- total_parsed_t100_windows: `11848`
- unique_hashes: `6`
- conversion_ready_now_count: `0`
- UCY legal conversion ready: `False`

## Candidate Integrity Rows

| path | sha256 | size | source identity | agents | frames | max track | t100 windows | target | legal status |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `UCY/zara02/obsmat.txt` | `0b29bdacd54b9712...` | 1208066 | `UCY::zara02::obsmat` | 204 | 1052 | 583 | 2095 | True | `terms_unverified_user_confirmation_required` |
| `UCY/zara01/obsmat.txt` | `bd999ebcfbc50682...` | 632141 | `UCY::zara01::obsmat` | 148 | 866 | 197 | 97 | True | `terms_unverified_user_confirmation_required` |
| `UCY/students01/students001.txt` | `a6d87f278d94136f...` | 879714 | `UCY::students01::students001` | 415 | 444 | 352 | 1949 | False | `terms_unverified_user_confirmation_required` |
| `UCY/students03/obsmat.txt` | `d9fd7048eab7d3d9...` | 2752596 | `UCY::students03::obsmat` | 428 | 540 | 539 | 3413 | False | `terms_unverified_user_confirmation_required` |
| `UCY/students03/obsmat_px.txt` | `8286c024198b7890...` | 2819811 | `UCY::students03::obsmat_px` | 428 | 540 | 540 | 3415 | False | `terms_unverified_user_confirmation_required` |
| `UCY/students03/students003.txt` | `e25798b660634330...` | 724279 | `UCY::students03::students003` | 434 | 541 | 289 | 879 | False | `terms_unverified_user_confirmation_required` |

## Interpretation

- This manifest locks lightweight file identity and parse stats for UCY h100 candidates.
- It stores hashes and counts only; it does not store raw trajectories.
- It does not grant legal permission and does not execute conversion/evaluation.
- `UCY|100` remains blocked until terms/source identity/local path are confirmed and guarded conversion/no-leakage/source-CV pass.
- `TrajNet|100` remains separately hard-blocked by missing long raw source support.

## Gate

| gate | pass |
| --- | ---: |
| `gw_input_verified` | True |
| `fq_input_verified` | True |
| `fs_input_verified` | True |
| `ucy_candidates_loaded` | True |
| `all_candidate_files_exist` | True |
| `target_family_candidates_present` | True |
| `hashes_computed` | True |
| `parsed_stats_computed` | True |
| `raw_content_not_stored` | True |
| `legal_blocker_preserved` | True |
| `no_conversion_ready_claim` | True |
| `no_download_conversion_eval` | True |
| `user_action_written` | True |
| `no_future_test_or_central_velocity_leakage` | True |
| `no_metric_seconds_overclaim` | True |
| `stage5c_false` | True |
| `smc_false` | True |
