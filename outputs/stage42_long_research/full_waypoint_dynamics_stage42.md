# Stage42-C Full-Waypoint Dynamics

- source: `fresh_run`
- generated_at_utc: `2026-05-26T23:52:30.763544+00:00`
- git_commit: `c11f73d`
- input_hash: `1d3fdc849dbf7708a7fd39bc0ff5412792726ae7d69fc2c643de8bab3f0a85f9`

## Claim Boundary

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 dataset-local / raw-frame 2.5D 多智能体轨迹世界状态模型。
- Stage42-C full-waypoint evaluation 使用 dataset-local raw-frame，不能写成 metric 或 seconds-level。
- future endpoints / future waypoints 只作为 loss/eval label，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Label Reconstruction / Training

- full trajectory label source: `fresh_run`
- full-waypoint checkpoint training sources: `cached_verified_checkpoints_fresh_eval`
- Stage42-C evaluation source: `fresh_run`
- best full-waypoint model: `full_trajectory_ensemble`
- training deployment decision: `candidate_full_trajectory_world_state_probe`

## Full-Waypoint Comparison

| candidate | source | rows | ADE all | ADE t50 | ADE t100 diag | ADE hard/failure | ADE easy degr | FDE all | FDE t50 | near-collision d005 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `strongest_floor_linear` | `fresh_run` | 55528 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `teacher_repair_linear_bridge` | `fresh_run` | 55528 | 0.2036 | 0.1312 | 0.1337 | 0.1966 | -0.1445 | 0.1914 | 0.1675 | 0.0000 |
| `m3w_neural_v1_composite_tail_linear_bridge` | `fresh_run` | 55528 | 0.2103 | 0.1365 | 0.1469 | 0.2038 | -0.1451 | 0.1982 | 0.1739 | -0.0039 |
| `ungated_endpoint_linear_bridge` | `fresh_run` | 55528 | 0.2966 | 0.2152 | 0.3592 | 0.3294 | 1.2459 | 0.3106 | 0.3135 | 0.0000 |
| `full_waypoint_transformer_protected` | `fresh_run` | 55528 | 0.1858 | 0.1480 | 0.2286 | 0.1952 | -0.0000 | 0.1938 | 0.2158 | 0.0086 |
| `ungated_full_waypoint_transformer` | `fresh_run` | 55528 | 0.2966 | 0.2152 | 0.3592 | 0.3294 | 1.2459 | 0.3106 | 0.3135 | 0.0000 |

## Cached-Verified Comparisons

- `endpoint_to_full_linear_bridge_domain_local`: domain-local endpoint neural dynamics through a linear waypoint bridge.
- `learned_waypoint_shape_bridge`: protected learned waypoint-shape residual; positive but small shape gain.
- `graph_interaction_group_consistency`: group/neighbor consistency protected comparison.

## Protected Full-Waypoint By Domain

| domain | rows | ADE all | ADE t50 | ADE t100 diag | hard/failure | easy degr | switch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ETH_UCY` | 25901 | 0.2198 | 0.1611 | 0.2325 | 0.2261 | -0.0000 | 0.3855 |
| `TrajNet` | 20087 | 0.2473 | 0.2188 | 0.3719 | 0.2658 | -0.0000 | 0.3174 |
| `UCY` | 9540 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.0000 | 0.0000 |

## Bootstrap CI For Protected Full-Waypoint ADE

| slice | low | mid | high | n |
| --- | ---: | ---: | ---: | ---: |
| `all` | 0.1823 | 0.1858 | 0.1898 | 55528 |
| `t50` | 0.1419 | 0.1480 | 0.1546 | 13689 |
| `t100` | 0.2204 | 0.2285 | 0.2373 | 9905 |
| `hard_failure` | 0.1911 | 0.1951 | 0.1995 | 41741 |

## Interpretation

- Stage42-C now compares actual reconstructed future waypoint labels, not only endpoint FDE.
- The protected full-waypoint sequence model is evaluated against endpoint-linear bridges and cached-verified learned-shape / graph-interaction evidence.
- Ungated full-waypoint and ungated endpoint variants remain diagnostic and are not deployable if easy safety fails.
- All results remain raw-frame dataset-local 2.5D. No metric, seconds-level, true 3D, Stage5C, or SMC claim is made.

## Gate Verdict

`stage42_c_full_waypoint_dynamics_pass` (12 / 12)
