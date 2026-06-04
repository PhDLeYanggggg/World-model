# Stage43-DK Scene/Graph Failure Taxonomy

- source: `fresh_stage43_dk_scene_graph_failure_taxonomy`
- result_source: `fresh_taxonomy_from_stage43_bl_ag_bo_bp_bq_dj`
- verdict: `stage43_dk_scene_graph_failure_taxonomy_pass_next_graph_first_moe`
- gate: `14 / 14`
- deployable policy changed: `False`
- next training contract: `stage43_next_graph_first_scene_residual_moe`

## What I Found

- Full graph is useful: t50 `30.65%`, hard/failure `36.93%`, easy `0.00%`.
- Geometry-route scene proxy is useful but narrow: t50 `35.32%`, easy `0.00%`.
- Full scene proxy is not deployment-safe: easy `9.59%`.
- Naive scene+graph full fusion underperforms graph-only: t50 delta `-11.30%`, easy delta `13.44%`.
- Learned gated fusion is safe but loses t50 vs best single: delta `-14.44%`.

## Failure Taxonomy

### naive_scene_graph_fusion_suppresses_graph_signal

- evidence: BP scene_graph_full t50 is 4.32%, while graph_history_only is 15.62%; delta -11.30%.
- interpretation: The strongest graph signal is real, but concatenating scene proxy and graph history changes the learned decision surface in a harmful way.
- repair target: Train graph-first mixture/routing where graph_history/full_graph is the default neural expert and scene proxy can only add residual context under a harm guard.

### scene_proxy_is_useful_but_not_raw_scene_or_sdf

- evidence: AG geometry_route improves t50 by 5.02% over no_scene with easy 0.00%, but full_scene easy is 9.59%.
- interpretation: Scene/goal proxy has signal, but broad scene proxy use can over-switch easy cases and still cannot support a raw-scene/SDF claim.
- repair target: Use a narrow geometry-route scene expert until raw scene/SDF tensors exist; keep raw-scene claims blocked.

### learned_gated_fusion_is_safe_but_too_conservative_or_misweighted

- evidence: BQ gated fusion loses 14.44% t50 versus the best single expert; bootstrap t50 contribution CI is [-33.68%, -29.90%].
- interpretation: The gate prevents the catastrophic easy damage of full fusion, but it also fails to preserve the graph expert's t50/hard lift.
- repair target: Add expert-preservation distillation: the fused model must not underperform graph_history_only on validation t50/hard before it can switch.

### t100_raw_frame_remains_diagnostic

- evidence: Full_graph t100 raw-frame diagnostic is -28.39%; BQ t100 delta versus best single is 2.99%.
- interpretation: The current scene/graph path is not a reliable long-horizon t100 solution.
- repair target: Keep t100 guarded and diagnostic; do not use it as deployment evidence until source/group support is stable.

## Next Training Contract

- name: `stage43_next_graph_first_scene_residual_moe`
- train next: `True`
- deployment rule: Default to the protected floor or graph expert; allow scene proxy residual only when gain is high, harm is low, and validation support says the source/horizon slice is covered.

Required experts:
- `no_context_floor_compatible_expert`
- `full_graph_or_graph_history_expert`
- `geometry_route_scene_proxy_expert`

Required losses:
- `full_waypoint_ade_loss`
- `failure_gain_harm_multitask_loss`
- `graph_expert_preservation_pairwise_loss`
- `easy_harm_penalty`
- `t50_hard_failure_weighting`
- `t100_guarded_diagnostic_reporting`

Do not repeat:
- `generic_scene_graph_full_concat`
- `gated_fusion_without_graph_expert_preservation`
- `raw_scene_claim_without_raw_scene_or_sdf_cache`

## Boundary

- This is a fresh evidence taxonomy, not a new deployed model.
- I keep the current protected latent-state candidate.
- No raw-scene/SDF claim until a raw-scene/SDF cache exists.
- Dataset-local/raw-frame 2.5D only; no metric/seconds, true-3D, foundation, Stage5C, or SMC claim.

## Gate

| gate | passed |
| --- | --- |
| `preconditions_passed` | `True` |
| `graph_signal_positive_and_easy_safe` | `True` |
| `scene_proxy_signal_present_but_guarded` | `True` |
| `naive_fusion_failure_identified` | `True` |
| `gated_fusion_safe_no_lift_identified` | `True` |
| `bootstrap_confirms_negative_gated_t50` | `True` |
| `next_training_contract_recorded` | `True` |
| `deployable_policy_not_changed` | `True` |
| `raw_scene_sdf_not_overclaimed` | `True` |
| `t100_remains_diagnostic` | `True` |
| `no_future_or_test_leakage` | `True` |
| `claim_boundary_not_overstated` | `True` |
| `stage5c_and_smc_false` | `True` |
| `long_objective_kept_active` | `True` |
