# Stage43-BQ Gated Scene-Graph Latent Fusion

- source: `fresh_stage43_bq_gated_scene_graph_fusion`
- result_source: `fresh_gated_scene_graph_latent_fusion`
- verdict: `stage43_bq_gated_scene_graph_fusion_pass_safe_no_lift_diagnostic`
- gate: `13 / 13`
- beats best single: `False`
- beats no context: `False`
- full multimodal unsafe: `False`
- deployable policy changed: `False`

## Protected Test Metrics

- all full-waypoint ADE improvement: `29.97%`
- t50 full-waypoint ADE improvement: `1.18%`
- hard/failure full-waypoint ADE improvement: `31.60%`
- easy degradation: `0.50%`
- switch rate: `42.58%`

## Contribution Deltas

- gated minus best single `graph_history_only` all/t50/hard: `-6.95%` / `-14.44%` / `-6.04%`
- gated minus no_context all/t50/hard: `-2.76%` / `-4.95%` / `-1.36%`

## Learned Context Gates

- scene gate mean: `0.1875`
- graph gate mean: `0.1909`
- scene gate easy/hard: `0.0890` / `0.2108`
- graph gate easy/hard: `0.0897` / `0.2119`

## Boundary

- This is a fresh retrained gated-fusion latent model, not inference masking.
- Scene inputs are train-only scene/goal/raster proxies, not raw images or verified metric SDF.
- Graph inputs are current-frame and past-only history graph summaries.
- Future waypoints are labels/eval only.
- It does not change the deployable protected policy.
- No metric, seconds-level, true-3D, foundation, Stage5C, or SMC claim.

## Gate

| gate | passed |
| --- | --- |
| `bp_negative_diagnostic_precondition_passed` | `True` |
| `fresh_gated_fusion_trained` | `True` |
| `not_raw_concat_or_inference_masking` | `True` |
| `scene_and_graph_dims_present` | `True` |
| `learned_gates_measured` | `True` |
| `latent_noncollapse` | `True` |
| `safe_easy_measured` | `True` |
| `best_single_comparison_reported` | `True` |
| `no_context_comparison_reported` | `True` |
| `checkpoints_not_committed` | `True` |
| `no_future_or_test_leakage` | `True` |
| `no_metric_seconds_stage5c_smc_claim` | `True` |
| `long_objective_kept_active` | `True` |
