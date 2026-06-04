# Stage43 Current World-Model Gate

- source: `fresh_stage43_dk_scene_graph_failure_taxonomy`
- verdict: `stage43_dk_scene_graph_failure_taxonomy_pass_next_graph_first_moe`
- passed: `14 / 14`
- protected multimodal latent-state candidate: `True`
- deployable policy changed: `False`
- next training contract: `stage43_next_graph_first_scene_residual_moe`
- long objective complete: `False`
- Stage5C executed: `False`
- SMC enabled: `False`

## Current Decision

Graph history has real signal, scene proxy has narrower signal, but naive scene+graph fusion and the current learned gate do not safely beat the graph expert. The next model should be graph-first with a scene residual expert and explicit expert-preservation loss.

## Key Evidence

- full_graph t50/hard/easy: `30.65%` / `36.93%` / `0.00%`
- geometry_route scene t50/easy: `35.32%` / `0.00%`
- scene_graph_full minus graph_history_only t50/easy: `-11.30%` / `13.44%`
- gated fusion minus best single t50: `-14.44%`

## Boundaries

- This does not change the deployable policy.
- Raw scene/SDF remains blocked.
- t100 remains raw-frame diagnostic.
- No metric/seconds, true-3D, foundation, Stage5C, or SMC claim.

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
