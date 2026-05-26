# Stage42 Ablation Tables

| ablation | source | status | all | t50 | hard/failure | easy | interpretation |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `no_neural_tail_use_teacher_floor_only` | `fresh_run` | `positive_safe` | 0.2036 | 0.1312 | 0.1966 | -0.1445 | Removing the composite-tail neural safe switch leaves the Stage37/teacher floor; positive protected-minus-teacher deltas indicate neural tail contribution. |
| `no_safe_floor_use_ungated_endpoint_neural` | `fresh_run` | `negative_unsafe` | 0.2966 | 0.2152 | 0.3294 | 1.2459 | Ungated endpoint neural is a no-fallback safety ablation; it can improve raw all but is not deployable if easy degradation exceeds 2%. |
| `oracle_floor_vs_neural_diagnostic` | `fresh_run` | `positive_safe` | 0.4222 | 0.3452 | 0.4211 | -0.2999 | Diagnostic oracle uses future labels only to measure remaining headroom; it is not a deployable model. |
| `no_full_waypoint_sequence_use_endpoint_linear_bridge` | `fresh_run` | `positive_safe` | 0.2103 | 0.1365 | 0.2038 | -0.1451 | Endpoint-linear bridge removes the full-waypoint sequence model. delta_vs_reference is ablation-minus-protected: negative t50/t100 deltas mean the full-waypoint model helps those horizons, while positive all-delta means endpoint-linear remains stronger on all-ADE. |
| `no_safe_floor_use_ungated_full_waypoint` | `fresh_run` | `negative_unsafe` | 0.2966 | 0.2152 | 0.3294 | 1.2459 | Ungated full-waypoint neural is a no-fallback safety ablation; it remains diagnostic if easy degradation is unsafe. |
| `no_composite_tail_use_teacher_linear_bridge` | `fresh_run` | `positive_safe` | 0.2036 | 0.1312 | 0.1966 | -0.1445 | Teacher linear bridge is the pre-composite floor in waypoint space; protected full-waypoint must improve without easy harm. |
| `no_history` | `cached_verified` | `complete` | n/a | n/a | n/a | n/a | Masks history/static causal feature group after policy freeze; proves coverage for no-history/static ablation, not a claim that every history-derived scalar is useless. |
| `no_neighbor` | `cached_verified` | `complete` | n/a | n/a | n/a | n/a | Neighbor/interaction masking is audited; group/neighbor features are especially important for the safety head and t100/hard slices. |
| `no_scene_goal` | `cached_verified` | `complete` | n/a | n/a | n/a | n/a | Scene/goal proxy coverage exists. Current deployable trajectory path keeps route/physical mostly diagnostic; route/physical heads are not main trajectory deployment claims. |
| `no_interaction` | `cached_verified` | `complete` | n/a | n/a | n/a | n/a | Interaction/group-consistency features have explicit ablations and are necessary for guarded deployment; without them raw neural remains less safe. |
| `no_jepa` | `cached_verified` | `complete_with_same_protocol_negative_evidence` | n/a | n/a | n/a | n/a | JEPA is explicitly disabled from the deployable path because audited JEPA variants were non-collapse but did not give deployable downstream lift. Stage41 now also records same-protocol pure-Transformer/no-JEPA attempts as negative or fallback-only, so the current positive path is protected endpoint neural dynamics rather than JEPA/Transformer purity. |
| `no_transformer` | `cached_verified` | `complete_with_same_protocol_negative_evidence` | n/a | n/a | n/a | n/a | Stage41 same-protocol JEPA-only/no-Transformer attempts are negative or unsafe, so no-Transformer is covered as negative architecture evidence. This is not a claim that JEPA contributes to the deployable path; it is why JEPA remains diagnostic-only. |
| `no_fallback` | `cached_verified` | `complete` | n/a | n/a | n/a | n/a | No-fallback neural often improves hard/all raw error but catastrophically damages easy cases; fallback is required for deployability. |

## Boundary

Stage42-D fresh-runs safety/floor/full-waypoint ablations and cached-verifies prior Stage30/41 component ablation evidence. It does not complete all-component retraining inside Stage42-D.

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

<!-- STAGE42_AC_REFRESH:START -->
## Stage42-AC Latest Evidence Refresh

- source: `fresh_synthesis_from_stage42_wxyz_aa_ab_artifacts`
- scope: protected dataset-local raw-frame 2.5D paper package only.
- Stage42-AB is now included as auxiliary-head evidence.
- Auxiliary-head conclusion: mixed / partial; not a uniform main contribution claim.
- Stage5C and SMC remain disabled.

| item | source | status | paper use | evidence |
| --- | --- | --- | --- | --- |
| Stage42-X unified row-level full-waypoint cache | `fresh_run_from_stage42s_row_cache_and_stage42v_ucy_predictions` | `stage42_x_unified_row_level_full_waypoint_cache_pass` | main protected 2.5D full-waypoint evidence | ADE all=0.090014, t50=0.061094, hard=0.093746, easy=0.001102 |
| Stage42-Y unified ablation evidence | `fresh_synthesis_from_stage42x_row_cache_and_retrained_ablation_reports` | `stage42_y_unified_ablation_evidence_pass` | paper-table ablation synthesis | gate=13/13; history/domain positive, goal/neighbor mixed |
| Stage42-Z claim evidence audit | `fresh_audit_from_stage42_wxy_and_paper_package_artifacts` | `stage42_z_paper_claim_evidence_audit_pass` | claim boundary audit | supports protected 2.5D raw-frame paper scope; rejects metric/seconds/foundation/ungated claims |
| Stage42-AA retrained ablation matrix | `fresh_matrix_from_stage42g_rerun_plus_stage42h_i_d_z` | `stage42_aa_retrained_ablation_matrix_pass_with_jepa_transformer_boundary` | required ablation coverage matrix | gate=15/15; no-JEPA cached negative; no-Transformer proxy boundary |
| Stage42-AB auxiliary-head retrained ablation | `fresh_run` | `stage42_ab_full_waypoint_auxiliary_ablation_pass` | mixed/partial auxiliary evidence, not main uniform-positive claim | no_aux all=-0.002339, t50=-0.037443; full-minus-no-aux t50=0.005361, all=-0.008219; uniform_positive=False |

### Auxiliary-Head Delta

| delta | value |
| --- | ---: |
| `ade_all_delta_full_minus_no_aux` | -0.008219 |
| `ade_t50_delta_full_minus_no_aux` | 0.005361 |
| `ade_hard_delta_full_minus_no_aux` | -0.009027 |
| `fde_t50_delta_full_minus_no_aux` | 0.005084 |
<!-- STAGE42_AC_REFRESH:END -->
