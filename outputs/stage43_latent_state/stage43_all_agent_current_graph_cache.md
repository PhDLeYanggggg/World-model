# Stage43-BM All-Agent Current Graph Cache

- source: `fresh_stage43_bm_all_agent_current_graph_cache`
- result_source: `fresh_build_current_frame_all_agent_knn_graph_cache_from_stage43_full_waypoint_rows`
- verdict: `stage43_bm_all_agent_current_graph_cache_pass_partial_history_blocker`
- gate: `14 / 14`
- all-agent current graph cache ready: `True`
- all-agent history graph cache ready: `False`
- raw scene/SDF cache ready: `False`

## Schema

- top_k: `8`
- group key: `['source_file', 'frame_id', 'horizon']`
- edge attrs: `['rel_x', 'rel_y', 'distance', 'inv_distance', 'bearing_cos', 'bearing_sin']`
- all_agent_history_xy: `not_available_in_stage43_full_waypoint_cache`

## Split Summary

| split | rows | groups | edges | multi-agent rows | mean degree | max agents/group | cache |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `train` | `146809` | `22657` | `896302` | `143706` | `6.105` | `65` | `data/stage43_all_agent_current_graph_cache/stage43_all_agent_current_graph_train.npz` |
| `val` | `101446` | `7911` | `749786` | `100748` | `7.391` | `65` | `data/stage43_all_agent_current_graph_cache/stage43_all_agent_current_graph_val.npz` |
| `test` | `89736` | `8278` | `630502` | `88199` | `7.026` | `58` | `data/stage43_all_agent_current_graph_cache/stage43_all_agent_current_graph_test.npz` |

## Validation

| split | row alignment | in range | no self edges | finite attrs | future labels absent |
| --- | --- | --- | --- | --- | --- |
| `train` | `True` | `True` | `True` | `True` | `True` |
| `val` | `True` | `True` | `True` | `True` | `True` |
| `test` | `True` | `True` | `True` | `True` | `True` |

## Boundary

- This is a current-frame all-agent neighbor graph cache, not an all-agent history graph cache.
- It does not include future endpoint or future waypoint inputs.
- It does not execute a retrained graph ablation yet.
- Raw-scene/SDF cache remains blocked.
- No metric, seconds-level, true-3D, foundation, Stage5C, or SMC claim.

## Gate

| gate | passed |
| --- | --- |
| `bl_precondition_passed` | `True` |
| `cache_files_written` | `True` |
| `train_val_test_rows_present` | `True` |
| `edge_tensors_present` | `True` |
| `all_agent_current_state_present` | `True` |
| `multi_agent_rows_present` | `True` |
| `row_alignment_preserved` | `True` |
| `edge_validation_passed` | `True` |
| `future_labels_not_in_inputs` | `True` |
| `current_graph_ready_but_history_graph_blocked` | `True` |
| `no_overclaim` | `True` |
| `no_future_or_test_leakage` | `True` |
| `stage5c_and_smc_false` | `True` |
| `long_objective_kept_active` | `True` |
