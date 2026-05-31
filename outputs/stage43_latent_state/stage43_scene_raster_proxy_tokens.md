# Stage43-AA Scene/Raster Proxy Tokens

- source: `fresh_stage43_aa_scene_raster_proxy_tokens`
- result_source: `fresh_train_only_scene_proxy_token_build_from_stage43_full_waypoint_cache`
- gate: `10 / 10`
- verdict: `stage43_aa_scene_raster_proxy_tokens_pass`
- scene/raster proxy token ready: `True`
- standalone deployment: `False`

## Build Summary

- build split: `stage43_train_only`
- source proxies: `10`
- domain proxies: `3`
- source proxy hash: `c56451af9f0be2fb8720e67111d7fb39e58014dbfc358794ff4e1e584acc1cc5`
- domain proxy hash: `42a0bbf40285dc0de6a7fbca8a719a8435dcc38e500ee040c65f4a82906317a8`
- manifest path: `data/stage43_scene_proxy_tokens/stage43_scene_proxy_manifest.json`
- manifest sha256: `87dc956e45a682b3a1c6bb8397e3c393745dc5f1ea44edb57a9d76f87693dfec`

## Split Features

| split | rows | feature dim | source coverage | source+domain coverage | feature hash |
| --- | ---: | ---: | ---: | ---: | --- |
| train | 146809 | 14 | 1.0000 | 1.0000 | `9f2b890a668b` |
| val | 101446 | 14 | 0.0000 | 1.0000 | `ed14f45e3a7b` |
| test | 89736 | 14 | 0.0000 | 1.0000 | `0f5ac09955b9` |

## Feature Names

- `scene_proxy_source_available`
- `scene_proxy_domain_available`
- `scene_proxy_level_source`
- `scene_proxy_rel_x`
- `scene_proxy_rel_y`
- `scene_proxy_boundary_sdf`
- `scene_proxy_route_occupancy`
- `scene_proxy_route_density_log`
- `scene_proxy_goal_dx_rel`
- `scene_proxy_goal_dy_rel`
- `scene_proxy_goal_alignment`
- `scene_proxy_entropy_mean`
- `scene_proxy_ambiguity_mean`
- `scene_proxy_rows_log`

## Boundary

- The proxy is built from Stage43 train rows only.
- It uses current/past positions, train route bounds, train route occupancy grids, domain/source route priors, and past-motion goal prototypes.
- It is not a raw scene image token, not an annotated walkable-area SDF, and not a metric scene map.
- Test endpoints and future waypoints are not used to build the proxy.
- Integration status: auxiliary cache ready; Stage43-M has not yet been retrained with these scene/raster proxy tokens.

## Gate

| gate | passed |
| --- | --- |
| stage43_z_precondition_passed | True |
| train_only_source_proxies_built | True |
| all_split_scene_proxy_features_built | True |
| row_hashes_recorded | True |
| source_or_domain_coverage_complete | True |
| scene_raster_proxy_features_present | True |
| goal_proxy_features_present | True |
| no_future_or_test_goal_leakage | True |
| claim_boundary_preserved | True |
| not_standalone_deployment | True |
