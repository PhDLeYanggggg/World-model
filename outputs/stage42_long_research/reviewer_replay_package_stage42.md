# Stage42-DM Reviewer Replay Package

- source: `fresh_reviewer_replay_package_from_stage42_runtime_and_manifest_artifacts`
- generated_at_utc: `2026-05-27T10:36:26.211498+00:00`
- git_commit: `27e5858`
- package_hash: `68bbb48d3b04c341eeb161c1bb54e9e873af7425b5c4059d2820c91330ae8b51`
- gate: `27 / 27`
- verdict: `stage42_dm_reviewer_replay_package_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-DM 是 reviewer replay package，不重新训练，不调 threshold。
- reviewer replay 只复现 freeze manifest、provenance、runtime replay 和 policy exact-replay 证据。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Replay Commands

- commands file: `outputs/stage42_long_research/reviewer_replay_commands_stage42.sh`

```bash
.venv-pytorch/bin/python run_stage42_replay_proximity_guard_policy.py
.venv-pytorch/bin/python run_stage42_batch_replay_proximity_guard_policy.py
.venv-pytorch/bin/python run_stage42_replay_group_consistency_policy.py
.venv-pytorch/bin/python run_stage42_group_consistency_runtime_policy.py
.venv-pytorch/bin/python run_stage42_module_contribution_ledger.py
.venv-pytorch/bin/python run_stage42_claim_boundary_linter.py
.venv-pytorch/bin/python run_stage42_source_action_consolidator.py
.venv-pytorch/bin/python run_stage42_evidence_provenance_verifier.py
.venv-pytorch/bin/python run_stage42_paper_freeze_candidate_manifest.py
.venv-pytorch/bin/python -m pytest tests/test_stage42_proximity_guard_policy_replay.py tests/test_stage42_proximity_guard_batch_replay.py tests/test_stage42_group_consistency_policy_replay.py tests/test_stage42_group_consistency_runtime_policy.py tests/test_stage42_module_contribution_ledger.py tests/test_stage42_claim_boundary_linter.py tests/test_stage42_source_action_consolidator.py tests/test_stage42_evidence_provenance_verifier.py tests/test_stage42_paper_freeze_candidate_manifest.py
```

## Evidence Inputs

| file | exists | sha256 |
| --- | --- | --- |
| `outputs/stage42_long_research/evidence_provenance_stage42.json` | `True` | `53238f8dfa86acacaac0e8a6c999b72dcc827c7fe20e4cefc231750ab034ef08` |
| `outputs/stage42_long_research/paper_freeze_candidate_manifest_stage42.json` | `True` | `e5e027c648b2d7fd41b2963a0c62f3bf0a092c6ec994d98b198110df2a01f6a9` |
| `outputs/stage42_long_research/proximity_guard_batch_replay_stage42.json` | `True` | `9e6cb21beba8e3f429113cb0eb6904c2f6ace55a3b7003411271eea7f9e6eb13` |
| `outputs/stage42_long_research/group_consistency_policy_replay_stage42.json` | `True` | `c5c6582d3b8d9bc60e806fb89a0050f1073a4daa46df8d3708df1e59ec390c33` |
| `outputs/stage42_long_research/group_consistency_runtime_policy_stage42.json` | `True` | `462d71d79a6797ddf6b44ad224fb46455b79da508944fcc8049b592d43d60305` |
| `outputs/stage42_long_research/module_contribution_ledger_stage42.json` | `True` | `66dc12358d61bfd497778ee3227d7753c3bb545a84404cab713a66f7c2c525c1` |
| `outputs/stage42_long_research/claim_boundary_linter_stage42.json` | `True` | `02507d984d4fd6de7d69017555ca2214304ffa50af5ca81280980a322795c064` |
| `outputs/stage42_long_research/source_action_consolidator_stage42.json` | `True` | `cf2187206c2364451f45dafaddd4865b77c544dad72c7b8a93ab49c8c624ec16` |

## Key Replay Metrics

- group runtime rows: `47458`
- switch exact match: `True`
- selected_xy_max_abs_diff: `0.0`
- selected_ade_max_abs_diff: `0.0`
- selected_fde_max_abs_diff: `0.0`
- all improvement: `0.24715658317833844`
- t50 improvement: `0.2236298792899738`
- t100 raw-frame diagnostic improvement: `0.1434611214781808`
- hard/failure improvement: `0.23887420070464105`
- easy degradation: `-0.2563085406508494`
- base near@0.05: `0.019364490707573012`
- final near@0.05: `0.01382274853554722`
- module ledger supported modules: `['history', 'domain_expert', 'safe_switch', 'teacher_floor', 'group_consistency_full_waypoint', 'full_waypoint_shape', 'endpoint_bridge']`
- blocked modules: `['scene_goal', 'neighbor_interaction', 'JEPA', 'Transformer']`
- claim linter violations: `0`
- source-action top actions: `['FW-TERMS-ucy_crowd_original', 'FW-H100-TrajNet|100', 'FW-DOMAIN-TrajNet', 'FW-DOMAIN-UCY', 'FW-H100-UCY|100']`
- source-action conversion_ready_now: `0`

## Gate

| gate | pass |
| --- | --- |
| `required_inputs_exist` | `True` |
| `commands_file_written` | `True` |
| `all_commands_use_arm64_venv` | `True` |
| `minimal_replay_has_no_training_commands` | `True` |
| `cx_provenance_passed` | `True` |
| `cz_manifest_passed` | `True` |
| `cv_batch_runtime_replay_passed` | `True` |
| `dk_group_policy_replay_passed` | `True` |
| `dl_runtime_policy_passed` | `True` |
| `paper_manifest_candidate_clean` | `True` |
| `manifest_hash_recorded` | `True` |
| `provenance_artifacts_count_ge_28` | `True` |
| `fu_module_ledger_passed` | `True` |
| `fv_claim_linter_passed` | `True` |
| `fv_claim_linter_zero_violations` | `True` |
| `fw_source_action_passed` | `True` |
| `fw_source_action_no_conversion_eval` | `True` |
| `fw_source_action_not_claim_ready` | `True` |
| `group_runtime_exact_replay` | `True` |
| `group_runtime_positive_all_t50_hard` | `True` |
| `group_runtime_t100_raw_reported` | `True` |
| `group_runtime_near_collision_reduced` | `True` |
| `no_metric_seconds_overclaim` | `True` |
| `true_3d_overclaim_blocked` | `True` |
| `foundation_overclaim_blocked` | `True` |
| `stage5c_false` | `True` |
| `smc_false` | `True` |

## Interpretation

- Stage42-DM gives reviewers a minimal deterministic replay path for the current Stage42 evidence package.
- It replays/freshens artifact checks and policy exact replay; it does not train, tune, or create new metric/seconds/3D/foundation claims.
- The supported claim remains protected dataset-local/raw-frame 2.5D multi-agent world-state evidence.
