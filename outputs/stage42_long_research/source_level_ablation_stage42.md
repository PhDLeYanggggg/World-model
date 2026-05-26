# Stage42-AN Proposed Source-Level Retrained Ablation

- source: `fresh_run`
- generated_at_utc: `2026-05-26T07:49:12.013468+00:00`
- git_commit: `0a32bce`
- input_hash: `7ecc6f74412b14912d33ada2a351461485b2d6bb6ff7b9b4063ffc55e8c3afc8`
- gate: `9 / 10`
- verdict: `stage42_an_source_level_ablation_partial_component_evidence`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-AN 是 proposed source-level split retrained ablation，不是 metric 或 seconds-level 结果。
- 每个特征消融都重新训练 ridge probe，并在 validation 上重新选 safe policy；test 只评一次。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Full Variant

- metrics: `{'rows': 47458, 'all_improvement': 0.24578838559866356, 't10_improvement': 0.4584029803090398, 't25_improvement': 0.23324144358920462, 't50_improvement': 0.22017071087030715, 't100_raw_frame_diagnostic_improvement': 0.14365240535176194, 'hard_failure_improvement': 0.23749387838492075, 'easy_degradation': -0.25662729975593146, 'switch_rate': 0.5880989506511021, 'harm_over_fallback': -0.11319535181660002}`
- feature_groups: `{'history': 121, 'neighbor_interaction': 5, 'goal_prototype': 10, 'baseline_family': 28, 'domain': 3, 'horizon': 4}`

## Retrained Variants

| variant | features | all | t50 | t100 diag | hard/failure | easy | switch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `full` | 166 | 0.245788 | 0.220171 | 0.143652 | 0.237494 | -0.256627 | 0.588099 |
| `no_history` | 45 | 0.262502 | 0.227640 | 0.142748 | 0.248648 | -0.301712 | 0.633739 |
| `no_neighbor_interaction` | 161 | 0.257361 | 0.223966 | 0.144255 | 0.240331 | -0.300654 | 0.621729 |
| `no_goal_prototype` | 156 | 0.258387 | 0.226508 | 0.143817 | 0.240185 | -0.313018 | 0.616651 |
| `no_baseline_family` | 138 | 0.178810 | 0.202283 | 0.000074 | 0.171817 | -0.151570 | 0.474757 |
| `no_domain_expert` | 163 | 0.245489 | 0.220681 | 0.142711 | 0.237193 | -0.254983 | 0.609360 |
| `motion_goal_no_baseline_domain` | 135 | 0.178755 | 0.203491 | -0.000517 | 0.171414 | -0.152355 | 0.479856 |

## Full Minus Ablated Variant

| ablation | delta all | delta t50 | delta hard/failure | interpretation |
| --- | ---: | ---: | ---: | --- |
| `no_history` | -0.016714 | -0.007470 | -0.011154 | Removing history does not hurt core metrics by > 0.01; contribution not proven by this ablation. |
| `no_neighbor_interaction` | -0.011573 | -0.003795 | -0.002837 | Removing neighbor_interaction does not hurt core metrics by > 0.01; contribution not proven by this ablation. |
| `no_goal_prototype` | -0.012598 | -0.006338 | -0.002691 | Removing goal_prototype does not hurt core metrics by > 0.01; contribution not proven by this ablation. |
| `no_baseline_family` | 0.066978 | 0.017888 | 0.065676 | Removing baseline_family_context hurts all_improvement, t50_improvement, hard_failure_improvement by > 0.01; contribution supported on this proposed source-level split. |
| `no_domain_expert` | 0.000299 | -0.000510 | 0.000301 | Removing domain_expert does not hurt core metrics by > 0.01; contribution not proven by this ablation. |
| `motion_goal_no_baseline_domain` | 0.067033 | 0.016680 | 0.066080 | Removing motion_goal_no_baseline_domain hurts all_improvement, t50_improvement, hard_failure_improvement by > 0.01; contribution supported on this proposed source-level split. |

## Safe Switch / Teacher Floor Analysis

- safe_switch_vs_ungated: `{'source': 'fresh_run', 'delta': {'protected_minus_ungated_all': -0.19762875904984278, 'protected_minus_ungated_t50': -0.18240007996739804, 'protected_minus_ungated_hard_failure': -0.19872304671312946, 'protected_minus_ungated_easy_degradation': 0.00944064090361596}, 'safe_switch_contribution_supported_here': False, 'interpretation': 'If protected trails ungated on accuracy but improves easy safety, safe-switch is safety-positive. If ungated is also safe, this specific ridge probe does not prove safe-switch necessity.'}`

## Interpretation

- Positive independent component evidence on this split: `['baseline_family_context']`.
- Positive combined/interacting ablation variants: `['motion_goal_no_baseline_domain']`. These do not count as independent modules by themselves.
- Stage42-AN does not prove two independent module contributions; next work should target stronger retrained neural/graph ablation or richer source-level features.
- This is retrained ridge ablation evidence on proposed source-level split. It does not prove true 3D, metric, seconds-level, foundation, Stage5C, or SMC readiness.
