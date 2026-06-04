# Stage44 WorldCore Architecture

M3W-WorldCore is a latent-state architecture for dataset-local/raw-frame 2.5D multi-agent world-state modeling.

## Token Schema

- `agent_history`: dim `103`
- `agent_state`: dim `2`
- `scene_image_raster`: dim `12`
- `walkable_obstacle`: dim `2`
- `goal_prototype`: dim `26`
- `interaction_edge`: dim `22`
- `time_source_domain_horizon`: dim `8`
- `baseline_rollout`: dim `18`

## Latent World State

`z_t = scene_latent + agent_latents + interaction_latents + goal_route_latent + occupancy_latent + uncertainty_latent`.

The implementation can save the token schema, replay deterministic tokens from caches, ablate token families through variant configs, and decode `z_next` through full-waypoint, endpoint, occupancy/density, interaction-risk, goal/route, safety, physical-validity, and uncertainty heads.

## Model Families

- No-baseline latent model: baseline rollout token disabled.
- Baseline-aware protected model: baseline context allowed, protected by floor policy.
- Hybrid JEPA + Transformer/SSM latent dynamics: future world-state latent target plus token dynamics.

Boundary: no Stage5C, no SMC, no metric/seconds/true-3D/foundation claim.
