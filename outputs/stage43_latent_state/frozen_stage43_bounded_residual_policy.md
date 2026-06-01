# Frozen Stage43 Bounded Residual Policy

- policy name: `stage43_bounded_residual_policy_v1`
- policy hash: `4dc482d146af2940b4968385e3f4bdf6951036b27b0353f1759a2130265ed493`
- policy type: `protected_bounded_residual_latent_waypoint`
- formula: `selected = floor_waypoint + alpha * clip_norm(neural_waypoint - floor_waypoint)`
- fallback: `floor waypoint`
- h100 guard: `True`

## Frozen Metrics

- all: `38.00%`
- t50: `26.96%`
- t100 diagnostic: `0.00%`
- hard/failure: `37.71%`
- easy degradation: `0.00%`
- t50 delta CI vs stored hard switch: `[9.86%, 11.21%]`

## Boundary

- Dataset-local/raw-frame 2.5D only.
- Future waypoints/endpoints are labels/eval only.
- No metric/seconds claim, no Stage5C, no SMC.
