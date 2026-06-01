# Stage43-BN All-Agent History Graph Cache

- source: `fresh_stage43_bn_all_agent_history_graph_cache`
- result_source: `fresh_build_past_only_all_agent_history_graph_cache_from_stage37_history_and_stage43_current_graph`
- verdict: `stage43_bn_all_agent_history_graph_cache_pass_raw_scene_blocker`
- gate: `13 / 13`
- all-agent history graph cache ready: `True`
- raw scene/SDF cache ready: `False`
- retrained graph ablation executed: `False`

## Schema

- history_k: `16`
- all_agent_history_xy: `[rows, top_k_plus_target, history_k, 2]`
- edge history attrs: `['shared_valid_count', 'neighbor_path_length', 'target_path_length', 'neighbor_mean_speed', 'target_mean_speed', 'neighbor_minus_target_mean_speed']`

## Split Summary

| split | rows | history_k | rows full target history | rows any neighbor history | edge count | cache |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `train` | `146809` | `16` | `40968` | `143706` | `896302` | `data/stage43_all_agent_history_graph_cache/stage43_all_agent_history_graph_train.npz` |
| `val` | `101446` | `16` | `14369` | `100748` | `749786` | `data/stage43_all_agent_history_graph_cache/stage43_all_agent_history_graph_val.npz` |
| `test` | `89736` | `16` | `52050` | `88199` | `630502` | `data/stage43_all_agent_history_graph_cache/stage43_all_agent_history_graph_test.npz` |

## Validation

| split | row alignment | shapes finite | future labels absent |
| --- | --- | --- | --- |
| `train` | `True` | `True` | `True` |
| `val` | `True` | `True` | `True` |
| `test` | `True` | `True` | `True` |

## Boundary

- This cache uses Stage37 past-only history windows and Stage43-BM current graph neighbors.
- Future endpoint/full-waypoint labels are not cached as inputs.
- This enables a future retrained graph ablation, but no graph ablation is executed here.
- Raw-scene/SDF cache remains blocked.
- No metric, seconds-level, true-3D, foundation, Stage5C, or SMC claim.

## Gate

| gate | passed |
| --- | --- |
| `bm_precondition_passed` | `True` |
| `history_cache_files_written` | `True` |
| `train_val_test_rows_present` | `True` |
| `target_history_present` | `True` |
| `neighbor_history_present` | `True` |
| `row_alignment_preserved` | `True` |
| `shape_and_finite_validation_passed` | `True` |
| `future_labels_not_in_history_graph_inputs` | `True` |
| `history_graph_ready_but_raw_scene_blocked` | `True` |
| `no_overclaim` | `True` |
| `no_future_or_test_leakage` | `True` |
| `stage5c_and_smc_false` | `True` |
| `long_objective_kept_active` | `True` |
