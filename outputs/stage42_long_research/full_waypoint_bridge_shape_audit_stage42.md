# Stage42-CM Full-Waypoint Bridge / Shape Audit

- source: `fresh_synthesis_from_stage42_full_waypoint_artifacts`
- generated_at_utc: `2026-05-26T18:27:51.830273+00:00`
- git_commit: `47068c7`
- input_hash: `00c666fd281936063e7f2d35e0e72546af9863ad3feb1dfd2096796b571c587a`
- gate: `14 / 14`
- verdict: `stage42_cm_full_waypoint_bridge_shape_audit_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-CM 是 endpoint/bridge/full-waypoint shape audit，不重新训练，不调 threshold，不执行 Stage5C，不启用 SMC。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- endpoint-only 成功不能自动算 full-waypoint world-state dynamics 成功。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。

## Key Deltas

### full_waypoint_minus_linear_bridge
- all_improvement: `-2.45%`
- t50_improvement: `1.15%`
- t100_raw_frame_diagnostic_improvement: `8.16%`
- hard_failure_improvement: `-0.87%`

### graph_group_minus_full_waypoint
- all_improvement: `3.66%`
- t50_improvement: `0.29%`
- t100_raw_frame_diagnostic_improvement: `0.16%`
- hard_failure_improvement: `2.89%`
- collision_delta_vs_floor_005: `0.83%`

## Comparison Rows

| variant | source | status | rows | all | t50 | t100 diag | hard | easy | switch | note |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `endpoint_only_final_fde` | `fresh_run` | `diagnostic_only_not_full_waypoint` | 55528 | 0.00% | 0.00% | 0.00% | 0.00% | -0.00% | 0.00% | Endpoint-only final FDE is not a full-waypoint world-state model. |
| `m3w_neural_v1_composite_tail_linear_bridge` | `fresh_run` | `deployable_endpoint_linear_bridge_floor` | 55528 | 21.03% | 13.65% | 14.69% | 20.38% | -14.51% | 34.10% | Current protected endpoint dynamics projected through a linear bridge. |
| `full_waypoint_transformer_protected` | `fresh_run` | `protected_full_waypoint_positive_two_domains` | 55528 | 18.58% | 14.80% | 22.86% | 19.52% | -0.00% | 29.46% | Actual full-waypoint sequence model under protected switch policy. |
| `ungated_full_waypoint_transformer` | `fresh_run` | `diagnostic_unsafe_not_deployable` | 55528 | 29.66% | 21.52% | 35.92% | 32.94% | 124.59% | 100.00% | Ungated neural/full-waypoint output has unsafe easy degradation. |
| `graph_interaction_group_consistency` | `cached_verified` | `protected_positive_with_proximity_caveat` | 55528 | 22.24% | 15.09% | 23.02% | 22.41% | 0.00% | 26.55% | Protected graph/group policy positive, but collision_delta_vs_floor_005=0.00829083972266037; CK blocks graph as independent main claim. |
| `unified_row_level_full_waypoint_cache` | `fresh_run_from_stage42s_row_cache_and_stage42v_ucy_predictions` | `row_level_full_waypoint_three_domain_positive` | 166584 | 9.00% | 6.11% | 8.15% | 9.37% | 0.11% | 23.26% | Unified row-level cache merges verified external full-waypoint policy sources across ETH_UCY, TrajNet, and UCY. |
| `ucy_endpoint_to_full_linear_bridge` | `fresh_run` | `failed_blocker` | n/a | n/a | n/a | n/a | n/a | n/a | n/a | Stage41 pure-UCY endpoint residual is positive on endpoint FDE, but linear endpoint-to-waypoint interpolation is negative on Stage42 full-waypoint validation and UCY test. Endpoint success cannot be counted as full-waypoint world-state success. |

## Interpretation

- Full-waypoint evidence exists and is strongest on horizon/full-waypoint slices, but endpoint-linear bridge remains stronger on all-ADE.
- Endpoint-only or endpoint-to-linear evidence must not be counted as learned full-waypoint shape by itself.
- Ungated full-waypoint neural remains unsafe; deployment still requires protected switch/floor.
- Graph/group interaction can help protected policy metrics, but CK blocks it as a standalone source-level graph main claim.
