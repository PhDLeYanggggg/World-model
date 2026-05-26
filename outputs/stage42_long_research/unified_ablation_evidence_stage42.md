# Stage42-Y Unified Ablation Evidence

- source: `fresh_synthesis_from_stage42x_row_cache_and_retrained_ablation_reports`
- generated_at_utc: `2026-05-26T04:46:01.553602+00:00`
- git_commit: `7c0e1b6`
- input_hash: `a88056df873cad45611d380338de6e50c0248cc6dcac457cffd11d52e422f0ae`
- gate: `13 / 13`
- verdict: `stage42_y_unified_ablation_evidence_pass`

## Current Facts
- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 dataset-local / raw-frame 2.5D 多智能体轨迹世界状态模型。
- Stage42-Y 汇总 row-level full-waypoint cache 与 retrained ablation evidence；不是 metric 或 seconds-level 结果。
- Stage42-X 统一 cache 是本轮 row-level full-waypoint 主证据；Stage42-H 是 retrained sequence ablation；Stage42-E 是 safety-floor 研究。
- future waypoints / endpoints 只作为 train/val labels 和 eval labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test 调阈值。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Stage42-X Reference

- ADE all: `0.090014`
- ADE t50: `0.061094`
- ADE t50 seed CI low: `0.053671`
- ADE hard/failure: `0.093746`
- easy degradation: `0.001102`

## Row-Level Full-Waypoint Ablation

| ablation | source | ADE all | ADE t50 | hard | easy | loss all | loss t50 | loss hard |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `floor_only` | `fresh_reference` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.090014 | 0.061094 | 0.093746 |
| `stage42j_static_expert_only` | `cached_verified_stage42r_row_cache` | 0.036222 | 0.036875 | 0.039705 | 0.000000 | 0.053792 | 0.024218 | 0.054040 |
| `stage42p_gain_harm_only` | `cached_verified_stage42r_row_cache` | 0.051537 | 0.006596 | 0.053256 | 0.008580 | 0.038477 | 0.054498 | 0.040490 |
| `stage42s_combo_no_ucy_source` | `cached_verified_stage42s_row_cache` | 0.052387 | 0.037934 | 0.054792 | 0.001102 | 0.037627 | 0.023159 | 0.038954 |
| `stage42x_unified_full` | `fresh_run_row_level_unified_cache` | 0.090014 | 0.061094 | 0.093746 | 0.001102 | 0.000000 | 0.000000 | 0.000000 |

## Retrained Sequence Ablation

| module | source | all | t50 | hard | full-minus-ablation all | t50 | hard |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `history tokens` | `fresh_run` | 0.330436 | 0.325545 | 0.337275 | 0.448036 | 0.457817 | 0.470799 |
| `domain expert` | `fresh_run` | 0.736930 | 0.741477 | 0.768207 | 0.041542 | 0.041885 | 0.039867 |
| `goal/scene tokens` | `fresh_run` | 0.773151 | 0.787621 | 0.803702 | 0.005321 | -0.004259 | 0.004372 |
| `neighbor/interaction tokens` | `fresh_run` | 0.778549 | 0.782064 | 0.809417 | -0.000078 | 0.001298 | -0.001343 |

## Safety Floor Evidence

| policy | all | t50 | hard | easy degradation | switch | deployable |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `floor_only` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | `True` |
| `ungated_endpoint_neural` | 0.296621 | 0.215203 | 0.329374 | 1.245861 | 1.000000 | `False` |
| `ungated_full_waypoint_neural` | 0.296621 | 0.215203 | 0.329374 | 1.245861 | 1.000000 | `False` |
| `teacher_raw_policy` | 0.351474 | 0.236664 | 0.350941 | 0.000000 | 0.461947 | `True` |
| `current_composite_tail_policy` | 0.210251 | 0.136522 | 0.203849 | 0.000000 | 0.341017 | `True` |

## Interpretation

- Stage42-X is now the unified row-level full-waypoint reference over ETH_UCY, TrajNet, and UCY.
- Removing the UCY full-waypoint source reverts to Stage42-S and loses t50/hard performance, so UCY source contribution is measurable.
- Retrained sequence ablation shows history tokens are the strongest proven sequence component; domain expert also contributes positively.
- Goal/scene and neighbor/interaction evidence is mixed in the current retrained sequence table and should not be overstated.
- Safety-floor evidence remains essential: ungated neural improves raw errors but is not deployable when easy degradation violates the gate.
- All claims remain raw-frame dataset-local 2.5D; Stage5C and SMC remain disabled.
