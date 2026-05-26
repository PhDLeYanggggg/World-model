# Stage42-DF All-Hard / Proximity Full-Waypoint Repair

- source: `fresh_stage42_df_all_hard_proximity_full_waypoint_repair`
- generated_at_utc: `2026-05-26T21:59:06.361246+00:00`
- git_commit: `41617c1`
- gate: `12 / 14`
- verdict: `stage42_df_all_hard_proximity_repair_partial`
- deployment_decision: `all_hard_proximity_repair_no_primary_promotion_keep_cq_guarded_composer`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-DF 针对 Stage42-DE 的 full-waypoint all/hard/proximity blocker，做 validation-only all+hard repair policy search。
- 本阶段重新评估 endpoint/full-waypoint common rows，不训练新模型，不用 test 调 threshold。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不能作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Validation-Selected Policy

- policy: `{'type': 'all_hard_proximity_full_waypoint_repair', 'min_all_gain': -0.005, 'min_hard_gain': -0.005, 'easy_max': 0.02, 'min_sep': 0.05, 'margin': 0.0}`
- selected_slices: `['ETH_UCY|50', 'ETH_UCY|100']`
- candidate_count: `73`

## Test Once vs Endpoint-Linear

- all: `-0.67%`
- t50: `-1.40%`
- t100 raw diagnostic: `-0.66%`
- hard/failure: `-0.72%`
- easy degradation: `0.19%`
- switch_rate: `13.16%`
- near_collision@0.05 delta vs endpoint: `-0.11%`

## Delta vs Stage42-CQ Guarded Composer

- delta_all: `-2.44%`
- delta_t50: `-2.46%`
- delta_t100_raw: `-4.14%`
- delta_hard: `-2.65%`
- delta_easy: `-0.06%`
- delta_near_collision@0.05: `-0.05%`

## Top Validation Candidates

| rank | selected slices | val all | val hard | val easy | val near@0.05 | eligible |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | `['ETH_UCY|50', 'ETH_UCY|100']` | 2.84% | 3.19% | 0.13% | 0.00% | `True` |
| 2 | `['ETH_UCY|50', 'ETH_UCY|100']` | 2.84% | 3.19% | 0.13% | 0.00% | `True` |
| 3 | `['ETH_UCY|50', 'ETH_UCY|100']` | 2.84% | 3.19% | 0.13% | 0.00% | `True` |
| 4 | `['ETH_UCY|50', 'ETH_UCY|100']` | 2.84% | 3.19% | 0.13% | 0.00% | `True` |
| 5 | `['ETH_UCY|50', 'ETH_UCY|100']` | 2.84% | 3.19% | 0.13% | 0.00% | `True` |
| 6 | `['ETH_UCY|50', 'ETH_UCY|100']` | 2.84% | 3.19% | 0.13% | 0.00% | `True` |
| 7 | `['ETH_UCY|50', 'ETH_UCY|100']` | 2.84% | 3.19% | 0.13% | 0.00% | `True` |
| 8 | `['ETH_UCY|50', 'ETH_UCY|100']` | 2.84% | 3.19% | 0.13% | 0.00% | `True` |
| 9 | `['ETH_UCY|50', 'ETH_UCY|100']` | 2.84% | 3.19% | 0.13% | 0.00% | `True` |
| 10 | `['ETH_UCY|50', 'ETH_UCY|100']` | 2.84% | 3.19% | 0.13% | 0.00% | `True` |

## Interpretation

- Stage42-DF specifically tests whether an all+hard+proximity validation objective can repair the full-waypoint deployment blocker identified in Stage42-DE.
- The result remains a protected evaluator over existing aligned endpoint/full-waypoint rows; it is not a newly trained all-agent full-waypoint model.
- If it improves Stage42-CQ, it becomes the next candidate target for actual full-waypoint training. If it does not, the correct next step is changing the model/loss, not repeating threshold search.

## Claim Boundary

- claim_boundary: `{'true_3d': False, 'foundation_world_model': False, 'global_metric_claim_allowed': False, 'global_seconds_claim_allowed': False, 'stage5c_executed': False, 'smc_enabled': False}`
