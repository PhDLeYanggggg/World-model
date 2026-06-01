# Stage43-BL Raw-Scene / Graph-Rich Ablation Readiness

- source: `fresh_stage43_bl_raw_scene_graph_ablation_readiness`
- result_source: `fresh_readiness_audit_from_stage43_proxy_ablation_and_cache_schema`
- verdict: `stage43_bl_raw_scene_graph_ablation_readiness_pass_blocker_documented`
- gate: `15 / 15`
- raw scene retrained ablation ready now: `False`
- graph-rich retrained ablation ready now: `False`
- raw scene / graph-rich main claim allowed: `False`

## Current Evidence

- scene proxy full-scene minus no-scene t50: `5.79%`
- scene proxy full-scene minus no-scene hard: `0.97%`
- full minus no-goal t50: `9.70%`
- full minus no-neighbor/interaction t50: `14.07%`
- interaction proxy head signal: `True`

## Readiness Decision

- proxy retrained ablation available: `True`
- raw scene retrained ablation ready now: `False`
- graph-rich retrained ablation ready now: `False`
- reason: Current scene/goal/interaction evidence is useful proxy evidence, but the cache lacks raw-scene/SDF tensors and graph-rich all-agent edge tensors needed for the requested retrained raw-scene/graph-rich ablation.

## Full-Waypoint Cache Schema

| split | rows | row geometry | future labels | raw scene keys | graph-rich keys |
| --- | ---: | --- | --- | --- | --- |
| `train` | `146809` | `True` | `True` | `[]` | `[]` |
| `val` | `101446` | `True` | `True` | `[]` | `[]` |
| `test` | `89736` | `True` | `True` | `[]` | `[]` |

## Claim Boundary

- Existing scene/goal evidence is train-only proxy evidence, not raw image/SDF evidence.
- Existing interaction evidence is scalar/proxy/future-label diagnostic evidence, not graph-rich all-agent dynamics.
- Future labels remain loss/eval only.
- No metric, seconds-level, true-3D, foundation, Stage5C, or SMC claim.

## Gate

| gate | passed |
| --- | --- |
| `bk_precondition_passed` | `True` |
| `scene_proxy_tokens_passed_and_proxy_only` | `True` |
| `scene_proxy_retrained_ablation_exists` | `True` |
| `feature_family_retrained_ablation_exists` | `True` |
| `multiseed_feature_family_confirmation_exists` | `True` |
| `interaction_proxy_diagnostic_exists` | `True` |
| `full_waypoint_cache_has_row_geometry_and_labels` | `True` |
| `raw_scene_tensor_missing_not_overclaimed` | `True` |
| `graph_rich_tensor_missing_not_overclaimed` | `True` |
| `blocker_matrix_records_required_next_artifacts` | `True` |
| `no_new_training_or_conversion` | `True` |
| `no_future_or_test_leakage` | `True` |
| `claim_boundary_not_overstated` | `True` |
| `stage5c_and_smc_false` | `True` |
| `long_objective_kept_active` | `True` |
