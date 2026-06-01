# Stage43-BO Graph-History Retrained Ablation

- source: `fresh_stage43_bo_graph_history_retrained_ablation`
- result_source: `fresh_retrained_graph_history_ablation`
- mode: `small`
- verdict: `stage43_bo_graph_history_retrained_ablation_pass_contribution_supported`
- gate: `14 / 14`
- graph-history contribution supported: `True`
- deployable policy changed: `False`

## Full Graph Minus No Graph

- all full-waypoint ADE contribution: `7.73%`
- t50 full-waypoint ADE contribution: `15.37%`
- hard/failure full-waypoint ADE contribution: `6.49%`
- t50 bootstrap contribution CI: `[13.84%, 16.88%]`

## Variants

| variant | graph features | all | t50 | hard | easy | full-minus-variant all | full-minus-variant t50 | full-minus-variant hard | latent var |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `no_graph` | `0` | `30.15%` | `15.28%` | `30.45%` | `13.94%` | `7.73%` | `15.37%` | `6.49%` | `0.3434` |
| `current_graph_only` | `7` | `37.18%` | `22.08%` | `38.01%` | `0.00%` | `0.69%` | `8.58%` | `-1.08%` | `0.3836` |
| `history_graph_only` | `10` | `30.39%` | `20.06%` | `29.81%` | `0.00%` | `7.49%` | `10.59%` | `7.13%` | `0.2997` |
| `full_graph` | `17` | `37.88%` | `30.65%` | `36.93%` | `0.00%` | `0.00%` | `0.00%` | `0.00%` | `0.3122` |

## Boundary

- This is a fresh retrained graph-feature ablation, not inference masking.
- Future waypoints are labels/eval only.
- It does not use raw scene/SDF tensors.
- It does not change the deployable protected policy.
- No metric, seconds-level, true-3D, foundation, Stage5C, or SMC claim.

## Gate

| gate | passed |
| --- | --- |
| `bn_precondition_passed` | `True` |
| `fresh_retrained_graph_variants` | `True` |
| `full_and_no_graph_retrained` | `True` |
| `current_and_history_graph_variants_retrained` | `True` |
| `graph_features_used_by_full_graph` | `True` |
| `no_graph_uses_no_graph_features` | `True` |
| `bootstrap_or_resampling_recorded` | `True` |
| `latent_noncollapse` | `True` |
| `easy_safety_reported` | `True` |
| `graph_contribution_measured` | `True` |
| `checkpoints_not_committed` | `True` |
| `no_future_or_test_leakage` | `True` |
| `no_metric_seconds_stage5c_smc_claim` | `True` |
| `long_objective_kept_active` | `True` |
