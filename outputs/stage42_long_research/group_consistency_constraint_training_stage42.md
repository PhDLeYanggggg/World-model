# Stage42-EU Group-Consistency Constraint Training

- source: `fresh_stage42_group_consistency_constraint_training`
- generated_at_utc: `2026-05-27T04:19:56.546863+00:00`
- git_commit: `9a28cac`
- input_hash: `d053315701d09458a1fc10d3d71907a8ef8ad976b78caa52ffd392647d7b1e7d`
- gate: `15 / 18`
- verdict: `stage42_eu_group_consistency_constraint_training_positive_not_promoted`
- decision: `group_constraint_training_not_enough_keep_stage42_di_or_cq_floor`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-EU 把 Stage42-ES/ET 支持的 source/frame/horizon group-consistency target 放进训练权重，再做 validation-selected protected full-waypoint evaluation。
- group-consistency weights 只来自 train/val/test row 的当前帧、past/context feature、predicted rollout geometry 和 source/frame/horizon group key；future waypoints 只作为 loss/eval labels。
- 不下载、不转换、不执行 Stage5C、不启用 SMC。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- validation 选择模型和 repair policy；test 只评一次。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。

## Selected Training Candidate

- variant: `group_unsafe_weighted`
- feature_mode: `stage42_am_features`
- lambda: `10.0`
- val_score: `1.835853`
- policy_slice_count: `8`
- mean_train_weight: `1.000000`
- max_train_weight: `5.447126`

## Test Once After Group Repair

| all | t50 | t100 raw | hard/failure | easy | switch | near base/final |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 22.81% | 22.35% | 12.68% | 21.97% | -23.91% | 56.17% | 1.88%/1.33% |

## Delta vs Prior

| prior | all | t50 | t100 raw | hard | easy |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Stage42-AM` | -1.76% | 0.34% | -1.69% | -1.78% | 1.75% |
| `Stage42-DH` | -2.69% | 0.22% | -1.66% | -1.77% | 5.32% |
| `Stage42-DI` | -1.90% | -0.01% | -1.67% | -1.91% | 1.72% |

## Validation Rows

| rank | variant | lambda | val score | val all | val t50 | val hard | val easy |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `group_unsafe_weighted` | 10.0 | 1.835853 | 44.10% | 48.06% | 43.70% | -27.70% |
| 2 | `group_unsafe_weighted` | 1.0 | 1.835116 | 44.07% | 48.05% | 43.68% | -27.17% |
| 3 | `group_unsafe_weighted` | 0.1 | 1.834538 | 44.05% | 48.04% | 43.66% | -27.00% |
| 4 | `group_unsafe_weighted` | 100.0 | 1.834206 | 44.07% | 48.02% | 43.68% | -27.74% |
| 5 | `group_easy_safe_weighted` | 100.0 | 1.823049 | 43.76% | 47.74% | 43.38% | -26.10% |
| 6 | `group_unsafe_hard_weighted` | 100.0 | 1.821411 | 43.76% | 47.72% | 43.36% | -26.58% |
| 7 | `group_unsafe_hard_weighted` | 10.0 | 1.820589 | 43.73% | 47.69% | 43.32% | -26.94% |
| 8 | `group_unsafe_hard_weighted` | 1.0 | 1.820057 | 43.71% | 47.68% | 43.30% | -26.43% |
| 9 | `group_unsafe_hard_weighted` | 0.1 | 1.819760 | 43.70% | 47.67% | 43.29% | -26.34% |
| 10 | `group_graph_density_weighted` | 1.0 | 1.810888 | 43.38% | 47.56% | 42.95% | -24.99% |
| 11 | `group_graph_density_weighted` | 10.0 | 1.810883 | 43.38% | 47.57% | 42.95% | -25.32% |
| 12 | `group_graph_density_weighted` | 100.0 | 1.810851 | 43.37% | 47.60% | 42.95% | -26.32% |

## Gate

| gate | pass |
| --- | ---: |
| `source_level_split_rebuilt` | True |
| `full_waypoint_labels_available` | True |
| `group_constraint_weights_built` | True |
| `training_candidates_run` | True |
| `validation_selected_training_model` | True |
| `validation_selected_group_repair` | True |
| `test_all_positive_vs_floor` | True |
| `test_t50_positive_vs_floor` | True |
| `test_hard_positive_vs_floor` | True |
| `easy_degradation_under_2pct` | True |
| `near005_repaired_vs_own_base` | True |
| `beats_stage42_am_all` | False |
| `beats_stage42_di_all` | False |
| `beats_stage42_di_hard` | False |
| `no_leakage_pass` | True |
| `no_metric_seconds_overclaim` | True |
| `stage5c_false` | True |
| `smc_false` | True |
