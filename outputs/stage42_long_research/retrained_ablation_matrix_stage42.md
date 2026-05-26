# Stage42-AA Retrained Ablation Matrix

- source: `fresh_matrix_from_stage42g_rerun_plus_stage42h_i_d_z`
- generated_at_utc: `2026-05-26T20:35:05.558081+00:00`
- git_commit: `16fed1c`
- input_hash: `a9ac45ad7d92c441683221ebd862c07e6f0baade5027ad4f93b0d92618effbe8`
- gate: `15 / 15`
- verdict: `stage42_aa_retrained_ablation_matrix_pass_with_jepa_transformer_boundary`
- fresh_required_coverage: `11 / 12`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 dataset-local / raw-frame 2.5D 多智能体轨迹世界状态模型。
- Stage42-AA 汇总 fresh retrained ablation matrix；不把 cached architecture negative evidence 伪装成 fresh retraining。
- future endpoints / waypoints 只作为 train/val labels 和 eval labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test 调阈值。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Required Ablation Matrix

| ablation | source | variant | status | main claim? | all | t50 | hard | delta all | delta t50 | delta hard |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `no_history` | `fresh_run` | `sequence_no_history_tokens` | `positive_contribution` | `True` | 0.330436 | 0.325545 | 0.337275 | 0.448036 | 0.457817 | 0.470799 |
| `no_neighbor` | `fresh_run` | `no_neighbor` | `positive_contribution` | `True` | 0.808752 | 0.846036 | 0.842339 | 0.003465 | 0.000106 | 0.003550 |
| `no_scene` | `fresh_run` | `no_scene_goal` | `positive_contribution` | `True` | 0.802580 | 0.846064 | 0.835035 | 0.009638 | 0.000078 | 0.010854 |
| `no_goal` | `fresh_run` | `no_goal` | `positive_contribution` | `True` | 0.802580 | 0.846064 | 0.835035 | 0.009638 | 0.000078 | 0.010854 |
| `no_interaction` | `fresh_run` | `no_interaction` | `positive_contribution` | `True` | 0.808752 | 0.846036 | 0.842339 | 0.003465 | 0.000106 | 0.003550 |
| `no_JEPA` | `cached_verified` | `jepa_only` | `cached_negative_or_inconclusive` | `False` | -0.026777 | -0.015066 | -0.024301 | 0.000000 | 0.000000 | 0.000000 |
| `no_Transformer` | `fresh_run` | `no_transformer_proxy_history_sequence` | `negative_or_inconclusive` | `False` | 0.817834 | 0.852847 | 0.859902 | -0.005617 | -0.006705 | -0.014014 |
| `no_teacher_floor` | `fresh_run` | `no_safe_floor_use_ungated_endpoint_neural` | `negative_unsafe` | `True` | 0.296621 | 0.215203 | 0.329374 | -0.086370 | -0.078681 | -0.125525 |
| `no_safe_switch` | `fresh_run` | `no_safe_switch` | `positive_contribution` | `True` | 0.790820 | 0.905293 | 0.801011 | 0.021397 | -0.059151 | 0.044877 |
| `no_endpoint_bridge` | `fresh_run` | `no_full_waypoint_sequence_use_endpoint_linear_bridge` | `positive_safe` | `True` | 0.210251 | 0.136522 | 0.203849 | -0.024473 | 0.011515 | -0.008669 |
| `no_full_waypoint_shape` | `fresh_run` | `no_full_waypoint_sequence_use_endpoint_linear_bridge` | `positive_safe` | `True` | 0.210251 | 0.136522 | 0.203849 | -0.024473 | 0.011515 | -0.008669 |
| `no_domain_expert` | `fresh_run` | `sequence_no_domain_expert` | `positive_contribution` | `True` | 0.736930 | 0.741477 | 0.768207 | 0.041542 | 0.041885 | 0.039867 |

## Interpretation

- Stage42-G was rerun and is the fresh retrained ridge ablation source for most required modules.
- Stage42-H supplies fresh causal sequence evidence; history tokens and domain expert are the cleanest positive sequence contributions.
- Stage42-I supplies full-waypoint secondary evidence but remains partial; it is not used to overclaim all full-waypoint ablation success.
- `no_JEPA` remains cached architecture evidence and negative/unsafe; it is included but not relabeled as fresh Stage42 retraining.
- `no_Transformer` has a fresh proxy ablation via history-sequence removal plus cached architecture evidence; this is not a full no-Transformer retrain claim.
- Removing teacher floor / safe floor is unsafe, so safety floor remains necessary.
- Claims remain dataset-local raw-frame 2.5D; Stage5C and SMC remain disabled.
