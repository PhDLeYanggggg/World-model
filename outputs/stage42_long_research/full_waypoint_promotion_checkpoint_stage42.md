# Stage42-DQ Full-Waypoint Promotion Checkpoint

- source: `fresh_synthesis_after_da3_full_waypoint_rerun`
- generated_at_utc: `2026-05-26T23:58:54.000718+00:00`
- git_commit: `c11f73d`
- gate: `24 / 24`
- verdict: `stage42_dq_full_waypoint_promotion_checkpoint_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-DQ 是 DA-3 fresh full-waypoint promotion checkpoint，不执行 Stage5C，不启用 SMC。
- 本阶段整合 fresh Stage42-C full-waypoint dynamics、fresh Stage42-CO common-validation composer、fresh Stage42-DI group-consistency repair、fresh Stage42-DL runtime replay。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。

## Fresh Evidence Chain

| stage | source | gate | verdict |
| --- | --- | --- | --- |
| `stage42_c_full_waypoint_dynamics` | `fresh_run` | `12/12` | `stage42_c_full_waypoint_dynamics_pass` |
| `stage42_co_common_validation_composer` | `fresh_common_validation_eval_from_cached_verified_checkpoints` | `14/14` | `stage42_co_common_validation_bridge_shape_composer_pass` |
| `stage42_di_group_consistency_repair` | `fresh_stage42_di_group_consistency_full_waypoint_repair` | `17/17` | `stage42_di_group_consistency_full_waypoint_repair_pass_promotable` |
| `stage42_dl_runtime_replay` | `fresh_runtime_api_from_frozen_group_consistency_policy_artifact` | `30/30` | `stage42_dl_group_consistency_runtime_policy_pass` |
| `stage42_dp_context_closure` | `fresh_synthesis_after_fresh_ar_as_rerun` | `n/a` | `stage42_dp_context_model_closure_pass` |

## Key Metrics

| policy | comparison floor | all | t50 | t100 raw diag | hard/failure | easy degradation |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `full_waypoint_transformer_protected` | `train_horizon_causal_floor` | 18.58% | 14.80% | 22.86% | 19.52% | -0.00% |
| `ungated_full_waypoint_transformer` | `train_horizon_causal_floor` | 29.66% | 21.52% | 35.92% | 32.94% | 124.59% |
| `common_validation_composer` | `endpoint_linear_bridge` | 3.02% | 1.50% | 6.12% | 3.28% | 0.25% |
| `group_consistency_repair` | `train_horizon_causal_floor` | 24.72% | 22.36% | 14.35% | 23.89% | -25.63% |
| `runtime_replay_group_consistency` | `train_horizon_causal_floor` | 24.72% | 22.36% | 14.35% | 23.89% | -25.63% |

## Runtime Replay And Safety

- switch_exact_match: `True`
- selected_xy_max_abs_diff: `0.0`
- selected_ade_max_abs_diff: `0.0`
- selected_fde_max_abs_diff: `0.0`
- runtime near@0.05 base/final/floor: `1.94%` / `1.38%` / `2.24%`

## Promotion Decision

- source_level_group_consistency_runtime_policy_promoted: `True`
- common_validation_endpoint_composer_remains_safety_sensitive_bridge: `True`
- global_primary_full_waypoint_replacement_claim_allowed: `False`
- ungated_full_waypoint_deployable: `False`
- reason: `Fresh DI/DL support a protected source-level group-consistency full-waypoint runtime policy with exact replay and proximity repair. However, common-validation endpoint-linear bridge/composer and source-level train-horizon floor use different comparison protocols, so the result cannot be collapsed into a single global primary full-waypoint replacement claim.`

## Next Best Action

- Keep Stage42-DL group-consistency runtime as source-level full-waypoint runtime evidence.
- Use Stage42-CQ proximity-aware composer as the safety-sensitive endpoint bridge/shape policy when endpoint-linear baseline is the comparison floor.
- Do not deploy ungated full-waypoint because easy degradation remains unsafe.
- Prioritize source/legal/time closure and protocol-aligned external sources before broader metric/seconds or global t100/full-waypoint claims.

## Claim Boundary

- Stage42-DQ supports a protected source-level full-waypoint runtime policy and exact replay evidence.
- It does not promote ungated full-waypoint dynamics.
- It does not collapse endpoint-linear composer and train-horizon source-level runtime into one global ranking.
- It remains dataset-local/raw-frame 2.5D; no metric/seconds-level, true 3D, foundation, Stage5C, or SMC claim is made.
