# Stage42-HW Replay Evidence Tier Refresh

- source: `fresh_replay_evidence_tier_refresh_from_stage42_hs_ht_hu_hv`
- generated_at_utc: `2026-05-27T20:37:18.682284+00:00`
- git_commit: `2888731`
- input_hash: `0a97309af73c139dd7e350a1aaa3b8ece0589c10ce3b1b7382e1482488b64a85`
- package_hash: `85683955feffc9f0de8564150ab7dd7589254b7dcdd390ae29a432865f72a20d`
- gate: `30 / 30`
- verdict: `stage42_hw_replay_evidence_tier_refresh_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-HW 把 HS/HT/HU/HV 的 replay 证据整理成 reviewer/paper evidence tiers。
- HW 不训练、不调阈值、不下载、不转换；它只刷新证据包和复现命令。
- HV row-level cache 是本地 derived artifact，不提交 GitHub。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Evidence Tiers

| tier | status | source | claim | rows | key metric | evidence |
| --- | --- | --- | --- | ---: | --- | --- |
| `T0_artifact_presence` | `pass` | `cached_verified_artifact_presence` | Required replay artifacts exist and are hashable. | 0 |  | outputs/stage42_long_research/group_consistency_t100_easy_guard_freeze_stage42.json<br>outputs/stage42_long_research/group_consistency_t100_easy_guard_runtime_stage42.json<br>outputs/stage42_long_research/t100_runtime_batch_replay_sufficiency_stage42.json<br>outputs/stage42_long_research/t100_runtime_row_cache_replay_stage42.json |
| `T1_runtime_smoke_replay` | `pass` | `fresh_runtime_api_from_frozen_stage42_hs_t100_easy_guard_policy` | Frozen t100 guard is callable and smoke-tested. | 4 |  | outputs/stage42_long_research/group_consistency_t100_easy_guard_runtime_stage42.json |
| `T2_frozen_metric_replay` | `pass` | `cached_verified_stage42_hr_policy_freeze_from_fresh_artifact` | Frozen policy decision table and metric summary replay exactly. | 0 | all 27.72%, t50 26.99%, t100raw 6.79%, hard 25.93% | outputs/stage42_long_research/group_consistency_t100_easy_guard_freeze_stage42.json |
| `T2_5_blocker_audit` | `resolved_by_hv` | `fresh_audit_from_stage42_hr_hs_ht_artifacts` | HU identified that HT smoke replay was insufficient for row-level batch replay; HV resolves this locally. | 0 | missing_row_level_candidate_floor_selected_arrays | outputs/stage42_long_research/t100_runtime_batch_replay_sufficiency_stage42.json<br>outputs/stage42_long_research/t100_runtime_row_cache_replay_stage42.json |
| `T3_row_level_batch_replay` | `pass` | `cached_verified_row_cache_runtime_batch_replay_from_stage42_hr_ht` | Frozen t100 runtime guard replayed over full row-level test cache with exact selected XY/ADE/switch/metric match. | 47458 | all 27.72%, t50 26.99%, t100raw 6.79%, hard 25.93% | outputs/stage42_long_research/t100_runtime_row_cache_replay_stage42.json |

## Replay Commands

- commands file: `outputs/stage42_long_research/reviewer_replay_commands_stage42_hv.sh`

```bash
.venv-pytorch/bin/python run_stage42_group_consistency_t100_easy_guard_runtime.py
.venv-pytorch/bin/python run_stage42_t100_runtime_batch_replay_sufficiency.py
.venv-pytorch/bin/python run_stage42_t100_runtime_row_cache_replay.py
.venv-pytorch/bin/python -m pytest tests/test_stage42_group_consistency_t100_easy_guard_runtime.py tests/test_stage42_t100_runtime_batch_replay_sufficiency.py tests/test_stage42_t100_runtime_row_cache_replay.py
```

## Gate

| gate | pass |
| --- | --- |
| `hs_artifact_passed` | `True` |
| `ht_runtime_smoke_passed` | `True` |
| `hu_blocker_audit_present` | `True` |
| `hv_row_level_replay_passed` | `True` |
| `tier0_present` | `True` |
| `tier1_present` | `True` |
| `tier2_present` | `True` |
| `tier25_blocker_resolved` | `True` |
| `tier3_row_level_present` | `True` |
| `row_level_rows_positive` | `True` |
| `cache_not_committed` | `True` |
| `row_level_metric_positive_all_t50_hard` | `True` |
| `t100_raw_reported_not_overclaimed` | `True` |
| `t100_easy_guard_preserved` | `True` |
| `replay_commands_written` | `True` |
| `commands_use_arm64_venv` | `True` |
| `commands_do_not_train_or_execute_forbidden` | `True` |
| `reviewer_package_updated` | `True` |
| `paper_matrix_updated` | `True` |
| `readmes_updated` | `True` |
| `no_future_endpoint_input` | `True` |
| `no_future_waypoint_input` | `True` |
| `no_central_velocity` | `True` |
| `no_test_endpoint_goals` | `True` |
| `no_test_threshold_tuning` | `True` |
| `no_metric_seconds_claim` | `True` |
| `true_3d_overclaim_blocked` | `True` |
| `foundation_overclaim_blocked` | `True` |
| `stage5c_not_executed` | `True` |
| `smc_not_enabled` | `True` |

## Interpretation

- Stage42-HW upgrades the paper/reviewer package from smoke/frozen replay evidence to explicit row-level batch replay evidence.
- HV cache remains local derived data and is deliberately not committed.
- The supported claim remains protected dataset-local/raw-frame 2.5D replay evidence, not metric/seconds-level, true 3D, Stage5C, or SMC.
