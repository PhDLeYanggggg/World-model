# Stage42-IN T50 Source-Specialist Reviewer Replay Package

- source: `cached_verified_stage42_ik_il_im_t50_source_specialist_reviewer_replay`
- generated_at_utc: `2026-05-28T01:59:58.189696+00:00`
- git_commit: `fa953c7`
- package_hash: `b7bc0ea17544d98ecb602174605e793fe159f9857d2cd11582120af2d786b52e`
- commands_file: `outputs/stage42_long_research/t50_source_specialist_replay_commands_stage42.sh`
- gate: `25 / 25`
- verdict: `stage42_in_t50_source_specialist_reviewer_replay_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-IN 是 t50 source-specialist reviewer replay package，不训练新模型，不调 threshold。
- IN 只把 Stage42-IK/IL/IM 的 source-specialist composition evidence 固化为可复现命令、hash 和 claim boundary。
- future waypoints / endpoints 只作为 supervised/evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Replay Commands

```bash
.venv-pytorch/bin/python run_stage42_t50_ensemble_ucy_specialist_integration.py
.venv-pytorch/bin/python run_stage42_t50_ucy_specialist_claim_audit.py
.venv-pytorch/bin/python run_stage42_t50_source_specialist_policy_freeze.py
.venv-pytorch/bin/python -m pytest tests/test_stage42_t50_ensemble_ucy_specialist_integration.py tests/test_stage42_t50_ucy_specialist_claim_audit.py tests/test_stage42_t50_source_specialist_policy_freeze.py tests/test_stage42_t50_source_specialist_reviewer_replay.py
```

## Required Files

| file | exists | sha256 |
| --- | --- | --- |
| `outputs/stage42_long_research/t50_ensemble_ucy_specialist_integration_stage42.json` | `True` | `03b773428d5075c3ac6775d55f26578d7474f9abc55ab79e89b3c559f794fdda` |
| `outputs/stage42_long_research/t50_ucy_specialist_claim_audit_stage42.json` | `True` | `8d5238f637ec8e0cc7293e33a364ace4fa047efe6b99e89e05f772476d5c6b48` |
| `outputs/stage42_long_research/t50_source_specialist_policy_freeze_stage42.json` | `True` | `6c93738aff15f40cabd4c46cf319acc3c2b7ab6e38668e77f23ce6cb3d9503d7` |
| `outputs/stage42_long_research/frozen_t50_source_specialist_policy_stage42.json` | `True` | `2a60a3b37f9681e2d2583255eccc13fa3d091e225735caf21eefd175d202366d` |

## Reviewer Replay Summary

- rows: `55528`
- ADE all / t50 / hard: `0.158819` / `0.104522` / `0.163730`
- ADE t50 CI low: `0.097328`
- ADE t100 raw-frame diagnostic: `0.180729`
- FDE t50 / CI low: `0.263687` / `0.256358`
- easy degradation: `0.000000`
- switch rate: `0.306440`
- UCY t50 before / after / delta: `0.000000` / `0.122892` / `0.122892`
- non-UCY max abs delta: `0.000000101979`
- policy hash: `9a73915ec3a74378def61d3a168f2d18c3fe0d6911fda1fcd971c8ada55ee1b2`

## Supported Claims

- {'claim': 'Stage42-IK repairs the Stage42-II/IJ UCY fallback-only t50 weak source under a row-aligned source specialist.', 'status': 'supported_fresh_composition_eval', 'evidence': 'UCY t50 0.000000 -> 0.122892; alignment rows 9540'}
- {'claim': 'The Stage42-II non-UCY ensemble decisions are unchanged by the IK composition.', 'status': 'supported_fresh_audit', 'evidence': 'max_abs_non_ucy_domain_metric_delta=0.000000101979'}
- {'claim': 'All powered t50 source files are nonnegative/positive after IK.', 'status': 'supported_fresh_audit', 'evidence': 'positive_powered_t50_source_count=3/3'}
- {'claim': 'IK improves the global Stage42-II ensemble while preserving easy cases.', 'status': 'supported_fresh_audit', 'evidence': 'all_delta=0.037627; t50_delta=0.023160; easy_degradation=0.000000'}

## Blocked Claims

- {'claim': 'IK proves a new independent external-domain generalization result.', 'status': 'blocked', 'reason': 'IK is a source-specialist composition using cached-verified row-aligned UCY full-waypoint branch evidence.'}
- {'claim': 'IK is new training.', 'status': 'blocked', 'reason': 'IK source labels mark new_training=not_run; it is a fresh composition/evaluation audit.'}
- {'claim': 'IK allows metric or seconds-level claims.', 'status': 'blocked', 'reason': 'claim boundary remains dataset-local/raw-frame only.'}
- {'claim': 'IK permits Stage5C or SMC.', 'status': 'blocked', 'reason': 'Stage5C and SMC remain false in all relevant artifacts.'}

## Gate

| gate | pass |
| --- | --- |
| `required_files_exist` | `True` |
| `required_files_hashed` | `True` |
| `commands_file_written` | `True` |
| `all_replay_commands_use_arm64_venv` | `True` |
| `no_training_or_threshold_search_commands` | `True` |
| `ik_gate_passed` | `True` |
| `il_gate_passed` | `True` |
| `im_gate_passed` | `True` |
| `im_compact_replay_exact` | `True` |
| `policy_hash_recorded` | `True` |
| `policy_artifact_hash_recorded` | `True` |
| `ucy_specialist_repair_positive` | `True` |
| `non_ucy_unchanged_with_tolerance` | `True` |
| `global_all_t50_hard_positive` | `True` |
| `t50_ci_low_positive` | `True` |
| `easy_preserved` | `True` |
| `no_future_or_test_leakage` | `True` |
| `reviewer_supported_claims_present` | `True` |
| `reviewer_blocked_claims_present` | `True` |
| `source_specialist_scope_only` | `True` |
| `no_metric_seconds_overclaim` | `True` |
| `true_3d_overclaim_blocked` | `True` |
| `foundation_overclaim_blocked` | `True` |
| `stage5c_false` | `True` |
| `smc_false` | `True` |

## Interpretation

- Stage42-IN turns the IK/IL/IM t50 source-specialist evidence into a compact reviewer replay package.
- It supports source-specialist composition and replay claims only; it is not new training or new independent-domain evidence.
- The result remains protected dataset-local/raw-frame 2.5D evidence, not metric, seconds-level, true 3D, foundation, Stage5C, or SMC evidence.
