# Stage42-JU Current Reviewer Replay Package

- source: `fresh_stage42_ju_current_reviewer_replay_package`
- generated_at_utc: `2026-05-29T05:51:12.698937+00:00`
- git_commit: `aa827ea`
- package_hash: `5d72b05cb70935420e26178c626ab621684ba0691412a43aad4f750d97ec172e`
- gate: `17 / 17`
- verdict: `stage42_ju_current_reviewer_replay_package_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-JU 是 current reviewer replay package，不重新训练、不调 threshold、不执行 Stage5C/SMC。
- 本包把当前 HEAD 的 IV/IW row-cache 正证据、AO incremental ablation 负证据、JS closure 和 JT claim refresh 串成可复现路径。
- future endpoints / waypoints 只允许作为 supervised/evaluation labels，不能作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。

## Replay Commands

- commands file: `outputs/stage42_long_research/current_reviewer_replay_commands_stage42.sh`

```bash
.venv-pytorch/bin/python run_stage42_source_level_row_cache_integration.py
.venv-pytorch/bin/python run_stage42_source_level_row_cache_mechanism_audit.py
.venv-pytorch/bin/python run_stage42_source_level_incremental_ablation.py
.venv-pytorch/bin/python run_stage42_source_context_gain_harm_closure.py
.venv-pytorch/bin/python run_stage42_current_module_claim_refresh.py
.venv-pytorch/bin/python -m pytest tests/test_stage42_source_level_row_cache_integration.py tests/test_stage42_source_level_row_cache_mechanism_audit.py tests/test_stage42_source_level_incremental_ablation.py tests/test_stage42_source_context_gain_harm_closure.py tests/test_stage42_current_module_claim_refresh.py tests/test_stage42_current_reviewer_replay_package.py
```

## Required Inputs

| file | exists | sha256 |
| --- | ---: | --- |
| `outputs/stage42_long_research/source_level_row_cache_integration_stage42.json` | `True` | `cc0fdd5ee1b0e0fbc30ce2b9ece2ef1ea40f9505bb3bcb0b94b00d58e0804e8e` |
| `outputs/stage42_long_research/source_level_row_cache_mechanism_audit_stage42.json` | `True` | `166b40bd3ea58b4f30d161c6832a33f6cef7b5a75279a834996f90430236c27a` |
| `outputs/stage42_long_research/source_level_incremental_ablation_stage42.json` | `True` | `66bcc01a0ef2ba91bb653c2e3c402ad8ddad2ea295189cfd52fc865c21bd6311` |
| `outputs/stage42_long_research/source_context_gain_harm_closure_stage42.json` | `True` | `395348ee46ee13760288e3cbe5533bee2f1a03b8deb50c6331091fd4991ed7aa` |
| `outputs/stage42_long_research/current_module_claim_refresh_stage42.json` | `True` | `2de8e5939c45a674e3e2660e65630bf8ea5921a4d4302b1afaedc38023423630` |

## Evidence Summary

- source domains: `{'TrajNet': 37918, 'UCY': 9540}`
- row-cache ADE all/t50/t100raw/hard: `0.291543` / `0.247045` / `0.196335` / `0.287273`
- easy degradation: `0.000000`
- t50 bootstrap CI: `[0.24292968809604645, 0.25138823240995406]`
- switch_rows: `33355`; fallback_exact_floor_rate: `1.0`
- AO positive standalone contexts: `['history_only', 'motion_goal_context']`
- AO positive incremental contexts after baseline-family: `[]`
- JS decision: `close_current_source_sequence_graph_gain_harm_family_for_t50_t100_main_claim`

## Allowed Claims

- protected source-level full-waypoint row-cache is positive on TrajNet+UCY under safe-switch/floor protection
- safe-switch and teacher/floor fallback are directly supported by row-cache mechanism evidence
- baseline-family rollout context remains the strongest current source-level driver
- history-only and motion-goal-context have standalone positive signal under AO, but only as bounded evidence

## Blocked Independent Claims

- incremental_context_after_baseline_family
- scene_goal_independent_main_claim
- neighbor_interaction_independent_main_claim
- sequence_graph_t50_t100_independent_main_claim
- JEPA_downstream_main_claim
- Transformer_independent_main_claim
- ungated_full_waypoint_deployment
- metric_seconds_or_true3d_claim

## Gate

| gate | pass |
| --- | ---: |
| `required_inputs_exist` | `True` |
| `commands_file_written` | `True` |
| `all_commands_use_arm64_venv` | `True` |
| `replay_commands_have_no_training_or_forbidden_execution` | `True` |
| `iv_row_cache_passed` | `True` |
| `iw_mechanism_passed` | `True` |
| `ao_negative_or_partial_recorded` | `True` |
| `js_closure_passed` | `True` |
| `jt_claim_refresh_passed` | `True` |
| `row_cache_positive_and_easy_safe` | `True` |
| `safe_switch_floor_replay_recorded` | `True` |
| `incremental_context_not_overclaimed` | `True` |
| `blocked_claims_include_context_and_neural` | `True` |
| `allowed_claims_are_protected_not_foundation` | `True` |
| `no_metric_seconds_or_3d_overclaim` | `True` |
| `stage5c_false` | `True` |
| `smc_false` | `True` |

## Interpretation

- Current replay supports protected source-level full-waypoint row-cache evidence with safe-switch/floor. It explicitly blocks independent scene/goal, neighbor/interaction, JEPA, Transformer, ungated, metric/time, true-3D and foundation claims.
- This package is for reviewer replay/provenance. It is not a new model training result and does not relax Stage42 claim boundaries.
