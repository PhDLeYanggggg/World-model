# Stage43-DL Graph-First Scene-Residual MoE

- source: `fresh_stage43_dl_graph_first_scene_residual_moe`
- result_source: `fresh_graph_first_scene_residual_moe`
- verdict: `stage43_dl_graph_first_scene_residual_moe_pass_safe_bq_lift_diagnostic`
- gate: `15 / 15`
- beats best single: `False`
- beats BQ gated fusion: `True`
- safe easy: `True`
- deployable policy changed: `False`

## Protected Test Metrics

- all full-waypoint ADE improvement: `30.34%`
- t50 full-waypoint ADE improvement: `7.78%`
- t100 raw-frame diagnostic: `-1.32%`
- hard/failure full-waypoint ADE improvement: `32.24%`
- easy degradation: `0.00%`
- switch rate: `41.40%`

## Contribution Deltas

- MoE minus best single `graph_history_only` all/t50/hard: `-6.58%` / `-7.84%` / `-5.40%`
- MoE minus BQ gated fusion all/t50/hard: `0.37%` / `6.60%` / `0.64%`

## Scene Residual Gate

- mean: `0.2942`
- easy mean: `0.0930`
- hard/failure mean: `0.3331`
- t50 mean: `0.2726`

## Boundary

- This is a fresh graph-first scene-residual MoE training run, not threshold-only tuning.
- Graph context is the default expert; scene proxy is only a gated residual expert.
- Scene evidence remains proxy scene/goal/raster evidence, not raw image/SDF evidence.
- Future waypoints are labels/eval only.
- It does not change the deployable protected policy.
- No metric, seconds-level, true-3D, foundation, Stage5C, or SMC claim.

## Gate

| gate | passed |
| --- | --- |
| `dk_next_training_contract_present` | `True` |
| `fresh_graph_first_moe_trained` | `True` |
| `graph_default_and_scene_residual_architecture` | `True` |
| `expert_preservation_loss_recorded` | `True` |
| `scene_and_graph_dims_present` | `True` |
| `latent_noncollapse` | `True` |
| `protected_eval_completed` | `True` |
| `easy_preservation_measured` | `True` |
| `best_single_comparison_reported` | `True` |
| `bq_comparison_reported` | `True` |
| `scene_residual_gate_measured` | `True` |
| `checkpoints_not_committed` | `True` |
| `no_future_or_test_leakage` | `True` |
| `no_metric_seconds_stage5c_smc_claim` | `True` |
| `long_objective_kept_active` | `True` |
