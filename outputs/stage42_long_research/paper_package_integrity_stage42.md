# Stage42-HX Paper Package Integrity

- source: `fresh_stage42_hx_paper_package_integrity_from_current_artifacts`
- generated_at_utc: `2026-05-27T20:47:04.074337+00:00`
- git_commit: `70a12a6`
- package_hash: `9ffbafc4608eb1d6daa7132338fb88731db487926328e04a2402438cfe3e4dad`
- gate: `25 / 25`
- verdict: `stage42_hx_paper_package_integrity_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-HX 是 paper package integrity/provenance verifier，不训练、不调阈值、不下载、不转换。
- paper package 可以支持 protected 2.5D world-state paper evidence，但不能支持 true-3D/foundation/global metric/seconds-level claim。
- future endpoints / waypoints 只作为 supervised labels 或 evaluation labels，不能作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Paper Deliverables

| path | role | source | size | sha256 |
| --- | --- | --- | ---: | --- |
| `outputs/stage42_long_research/paper_outline_stage42.md` | `paper_deliverable` | `cached_verified` | 47729 | `3220576babb3` |
| `outputs/stage42_long_research/method_draft_stage42.md` | `paper_deliverable` | `cached_verified` | 55082 | `fb13ee4a4a2f` |
| `outputs/stage42_long_research/experiment_tables_stage42.md` | `paper_deliverable` | `cached_verified` | 63952 | `a0b74751b15b` |
| `outputs/stage42_long_research/ablation_tables_stage42.md` | `paper_deliverable` | `cached_verified` | 70602 | `11886602530a` |
| `outputs/stage42_long_research/failure_taxonomy_stage42.md` | `paper_deliverable` | `cached_verified` | 50135 | `bcdce188a12d` |
| `outputs/stage42_long_research/model_card_stage42.md` | `paper_deliverable` | `cached_verified` | 60809 | `39f8260347cc` |
| `outputs/stage42_long_research/data_card_stage42.md` | `paper_deliverable` | `cached_verified` | 49368 | `83c17745349c` |
| `outputs/stage42_long_research/reproducibility_stage42.md` | `paper_deliverable` | `cached_verified` | 55155 | `6c1dc325e899` |
| `outputs/stage42_long_research/a_journal_gap_stage42.md` | `paper_deliverable` | `cached_verified` | 92463 | `674624d31ab2` |

## Support Evidence

| path | role | source | size | sha256 |
| --- | --- | --- | ---: | --- |
| `outputs/stage42_long_research/paper_ready_evidence_matrix_stage42.md` | `support_evidence` | `cached_verified` | 12979 | `a236125f2ce3` |
| `outputs/stage42_long_research/replay_evidence_tiers_stage42.md` | `support_evidence` | `cached_verified` | 5371 | `9c96f5a537ed` |
| `outputs/stage42_long_research/replay_evidence_tiers_stage42.json` | `support_evidence` | `cached_verified` | 72075 | `a1c5f6ffdad0` |
| `outputs/stage42_long_research/reviewer_replay_package_stage42.md` | `support_evidence` | `cached_verified` | 8499 | `4e825fd99e6f` |
| `outputs/stage42_long_research/reviewer_replay_commands_stage42_hv.sh` | `support_evidence` | `cached_verified` | 646 | `fc2587b25a8b` |
| `outputs/stage42_long_research/data_calibration_stage42.json` | `support_evidence` | `cached_verified` | 78134 | `0b549204e900` |
| `outputs/stage42_long_research/source_time_geometry_calibration_stage42.json` | `support_evidence` | `cached_verified` | 15612 | `070bb33f20b6` |
| `outputs/stage42_long_research/source_terms_validation_stage42.json` | `support_evidence` | `cached_verified` | 15040 | `21d06e155727` |
| `outputs/stage42_long_research/source_terms_gap_audit_stage42.json` | `support_evidence` | `cached_verified` | 13092 | `b89b22eec587` |
| `outputs/stage42_long_research/restricted_metric_time_readiness_stage42.json` | `support_evidence` | `cached_verified` | 14576 | `fffdcea71df7` |

## A-F Objective Coverage

| objective | source | present | expected status | status preserved |
| --- | --- | --- | --- | --- |
| `A_data_and_calibration` | `cached_verified` | `True` | `partial_blocked` | `True` |
| `A_metric_time_calibration` | `cached_verified` | `True` | `partial_blocked` | `True` |
| `B_external_validation` | `cached_verified` | `True` | `pass_with_boundary` | `True` |
| `C_full_waypoint_dynamics` | `cached_verified` | `True` | `pass_with_boundary` | `True` |
| `D_causal_ablation` | `cached_verified` | `True` | `mixed` | `True` |
| `E_safety_floor` | `cached_verified` | `True` | `pass_with_boundary` | `True` |
| `F_paper_package` | `cached_verified` | `True` | `pass_with_open_gaps` | `True` |

## Readiness Summary

- `data_calibration_source`: `fresh_run`
- `global_metric_claim_allowed`: `False`
- `global_seconds_claim_allowed`: `False`
- `source_terms_status`: `{'source': 'fresh_stage42_cg_source_terms_confirmation_validator', 'targets_validated': 5, 'terms_accepted_targets': 0, 'conversion_ready_targets': 0, 'conversion_allowed_now_count': 0, 'converted_datasets_now': 0, 'evaluated_datasets_now': 0, 'stage5c_executed': False, 'smc_enabled': False}`
- `source_terms_gap_status`: `fresh_rerun_cg_plus_ed_source_terms_gap_audit`
- `restricted_metric_time_ready`: `False`
- `restricted_metric_time_verdict`: `fresh_stage42_hi_restricted_metric_time_readiness`

## Gate

| gate | pass |
| --- | --- |
| `paper_files_exist` | `True` |
| `paper_files_nonempty` | `True` |
| `paper_files_hashable` | `True` |
| `support_files_exist` | `True` |
| `support_files_hashable` | `True` |
| `objective_a_to_f_rows_present` | `True` |
| `objective_statuses_preserved` | `True` |
| `data_calibration_blocker_preserved` | `True` |
| `metric_time_blocker_preserved` | `True` |
| `ablation_mixed_status_preserved` | `True` |
| `paper_package_open_gaps_preserved` | `True` |
| `stage42_hw_t3_replay_present` | `True` |
| `reviewer_replay_commands_include_hv` | `True` |
| `claim_boundary_not_true3d` | `True` |
| `claim_boundary_not_foundation` | `True` |
| `claim_boundary_not_metric_seconds` | `True` |
| `stage5c_not_executed` | `True` |
| `smc_not_enabled` | `True` |
| `future_endpoint_input_blocked` | `True` |
| `central_velocity_blocked` | `True` |
| `test_endpoint_goals_blocked` | `True` |
| `test_threshold_tuning_blocked` | `True` |
| `result_sources_labeled` | `True` |
| `no_raw_data_or_cache_in_manifest` | `True` |
| `readmes_updated` | `True` |

## Interpretation

- Stage42-HX verifies that the paper package is internally consistent and hashable.
- The package is paper-ready for protected raw-frame/dataset-local 2.5D evidence, while preserving open blockers for source terms, metric/time calibration, and mixed/negative module evidence.
- This is not new training, not new evaluation, not metric conversion, not Stage5C, and not SMC.
