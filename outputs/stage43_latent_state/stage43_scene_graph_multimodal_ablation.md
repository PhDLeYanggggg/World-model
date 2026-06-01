# Stage43-BP Scene-Graph Multimodal Retrained Ablation

- source: `fresh_stage43_bp_scene_graph_multimodal_ablation`
- result_source: `fresh_retrained_scene_graph_multimodal_ablation`
- mode: `small`
- verdict: `stage43_bp_scene_graph_multimodal_ablation_pass_negative_unsafe_diagnostic`
- gate: `16 / 16`
- multimodal contribution supported: `True`
- best-single lift supported: `False`
- full multimodal unsafe: `True`
- deployable policy changed: `False`

## Scene Graph Full Minus No Context

- all full-waypoint ADE contribution: `-1.03%`
- t50 full-waypoint ADE contribution: `-1.81%`
- hard/failure full-waypoint ADE contribution: `1.19%`
- t50 bootstrap contribution CI: `[-3.03%, -0.60%]`

## Scene Graph Full Minus Best Single By T50

- best single by t50: `graph_history_only`
- all contribution: `-5.22%`
- t50 contribution: `-11.30%`
- hard/failure contribution: `-3.49%`
- t50 bootstrap contribution CI: `[-12.44%, -10.22%]`

## Variants

| variant | scene | graph | all | t50 | hard | easy | full-minus-variant all | full-minus-variant t50 | full-minus-variant hard | latent var |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `no_context` | `0` | `0` | `32.73%` | `6.13%` | `32.96%` | `0.00%` | `-1.03%` | `-1.81%` | `1.19%` | `0.3434` |
| `scene_proxy_only` | `14` | `0` | `33.67%` | `3.63%` | `34.85%` | `0.00%` | `-1.98%` | `0.68%` | `-0.70%` | `0.3284` |
| `graph_history_only` | `0` | `17` | `36.91%` | `15.62%` | `37.64%` | `0.00%` | `-5.22%` | `-11.30%` | `-3.49%` | `0.3113` |
| `scene_graph_full` | `14` | `17` | `31.70%` | `4.32%` | `34.15%` | `13.44%` | `0.00%` | `0.00%` | `0.00%` | `0.3746` |

## Boundary

- This is a fresh retrained multimodal context ablation, not inference masking.
- Scene inputs are train-only scene/goal/raster proxies, not raw scene images or verified metric SDF.
- Graph inputs are current-frame and past-only history graph summaries.
- Future waypoints are labels/eval only.
- It does not change the deployable protected policy.
- No metric, seconds-level, true-3D, foundation, Stage5C, or SMC claim.

## Gate

| gate | passed |
| --- | --- |
| `scene_proxy_cache_precondition_passed` | `True` |
| `scene_proxy_retrained_ablation_precondition_passed` | `True` |
| `graph_history_retrained_ablation_precondition_passed` | `True` |
| `fresh_retrained_multimodal_variants` | `True` |
| `no_scene_no_graph_baseline_retrained` | `True` |
| `scene_only_and_graph_only_retrained` | `True` |
| `full_scene_graph_uses_both_modalities` | `True` |
| `bootstrap_recorded` | `True` |
| `latent_noncollapse` | `True` |
| `multimodal_beats_no_context_on_t50_or_hard` | `True` |
| `multimodal_best_single_comparison_reported` | `True` |
| `easy_safety_measured` | `True` |
| `checkpoints_not_committed` | `True` |
| `no_future_or_test_leakage` | `True` |
| `no_metric_seconds_stage5c_smc_claim` | `True` |
| `long_objective_kept_active` | `True` |
