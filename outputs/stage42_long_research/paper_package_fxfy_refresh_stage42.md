# Stage42-FZ Paper Package FX/FY Refresh

- source: `fresh_stage42_paper_package_fxfy_refresh`
- generated_at_utc: `2026-05-27T11:09:31.973663+00:00`
- git_commit: `a35d38e`
- input_hash: `e28d3e64c5fa38e7f4cf6ddb442375d3f94d066d5033b991f62ee77b63694532`
- gate: `20 / 20`
- verdict: `stage42_fz_paper_package_fxfy_refresh_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- FZ 只刷新 paper package；不训练、不下载、不转换、不调 test threshold。
- Stage5C latent generative 未执行；SMC 未启用。

## Summary

- supported_core_claims: `['history', 'domain_expert', 'safe_switch', 'teacher_floor', 'group_consistency_full_waypoint']`
- blocked_main_claims: `['JEPA_downstream_lift', 'ungated_neural_dynamics', 'scene_goal_independent_main_claim', 'neighbor_interaction_independent_main_claim', 'global_metric_seconds_claim']`
- objective_status_counts: `{'blocked_user_action_required': 1, 'partial_positive_with_source_blockers': 1, 'partial_protected_not_ungated': 1, 'partial_main_modules_identified': 1, 'pass_floor_required': 1, 'paper_package_candidate_clean_with_open_blockers': 1}`
- blocked_objectives: `['A']`
- partial_objectives: `['B', 'C', 'D']`
- passed_objectives: `['E']`
- weak_horizons: `['TrajNet|100', 'UCY|100']`
- stop_repeat_modeling_now: `True`
- uniform_horizon_claim_allowed: `False`
- highest_priority_next_action: `FW-TERMS-ucy_crowd_original`

## Paper Files

| file | exists | contains FZ marker | size bytes |
| --- | ---: | ---: | ---: |
| `outputs/stage42_long_research/paper_outline_stage42.md` | True | True | 47729 |
| `outputs/stage42_long_research/method_draft_stage42.md` | True | True | 51251 |
| `outputs/stage42_long_research/experiment_tables_stage42.md` | True | True | 61829 |
| `outputs/stage42_long_research/ablation_tables_stage42.md` | True | True | 69290 |
| `outputs/stage42_long_research/failure_taxonomy_stage42.md` | True | True | 48823 |
| `outputs/stage42_long_research/model_card_stage42.md` | True | True | 59497 |
| `outputs/stage42_long_research/data_card_stage42.md` | True | True | 45537 |
| `outputs/stage42_long_research/reproducibility_stage42.md` | True | True | 53843 |
| `outputs/stage42_long_research/a_journal_gap_stage42.md` | True | True | 78255 |

## Gate

| gate | pass |
| --- | ---: |
| `source_fresh` | True |
| `fu_gate_passed` | True |
| `fv_gate_passed` | True |
| `fw_gate_passed` | True |
| `dm_gate_passed` | True |
| `fx_gate_passed` | True |
| `fy_gate_passed` | True |
| `all_paper_files_refreshed` | True |
| `goal_not_marked_complete` | True |
| `data_blocker_preserved` | True |
| `weak_horizon_blocker_preserved` | True |
| `stop_repeat_retry_recorded` | True |
| `uniform_horizon_not_claimed` | True |
| `no_overclaim_violations` | True |
| `no_download_conversion_training_eval` | True |
| `no_test_threshold_tuning` | True |
| `no_metric_seconds_overclaim` | True |
| `no_true3d_foundation_overclaim` | True |
| `stage5c_false` | True |
| `smc_false` | True |

## Interpretation

- FX/FY are now explicitly represented in the paper package.
- The long goal remains active and incomplete; objective A remains blocked by source/legal conversion support.
- Uniform horizon robustness remains blocked by TrajNet|100 and UCY|100; repeating the same-feature model retry is explicitly discouraged.
