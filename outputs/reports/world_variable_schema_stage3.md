# Stage 3 Variable Schema

The model should stop treating trajectories as only `x/y/vx/vy`. Stage 3 expands each state into kinematic, interaction, scene, intent, observation, and uncertainty variables.

## Kinematic

| Variable | Meaning | Source |
| --- | --- | --- |
| `speed_mps` | agent speed in meters per second | observed or finite-difference |
| `acceleration_mps2` | agent acceleration in meters per second squared | observed in TGSIM or finite-difference |
| `jerk_mps3` | temporal change in acceleration, used for smoothness | derived |
| `heading_rate_radps` | turning rate | derived |
| `stopping_probability_proxy` | short-horizon low-speed/stall tendency | learned/derived |

## Interaction

| Variable | Meaning | Source |
| --- | --- | --- |
| `time_to_collision_s` | pairwise TTC under current relative velocity | derived |
| `closing_speed_mps` | relative speed along neighbor normal | derived |
| `front_density_people_per_m2` | density in front sector | derived |
| `rear_density_people_per_m2` | density behind sector | derived |
| `side_clearance_m` | minimum left/right clearance | derived from neighbors/obstacles |

## Scene

| Variable | Meaning | Source |
| --- | --- | --- |
| `region_id` | polygon/semantic region assignment | TGSIM polygons/OpenDD maps/manual scene geometry |
| `lane_or_area_type` | crosswalk, sidewalk, intersection, lane, plaza, obstacle | scene geometry |
| `distance_to_crosswalk_m` | metric distance to nearest crosswalk region | scene geometry |
| `distance_to_exit_m` | distance to candidate exit/goal | scene geometry |
| `bottleneck_score` | local corridor/narrow-passage score | derived from scene geometry |

## Intent

| Variable | Meaning | Source |
| --- | --- | --- |
| `goal_region_distribution` | probability over candidate exits/goals | latent inferred |
| `velocity_goal_alignment` | alignment between velocity and sampled goal | derived |
| `route_choice_entropy` | uncertainty over destination/route | latent inferred |
| `dwell_time_frames` | how long an agent has been stopped or near-stopped | derived |
| `intent_change_flag` | detected/sampled goal change | latent/event |

## Uncertainty

| Variable | Meaning | Source |
| --- | --- | --- |
| `observation_noise_sigma_m` | measurement noise in world units | dataset/calibration |
| `track_age_frames` | age of current identity track | tracking metadata |
| `missing_observation_count` | recent missing frames | tracking metadata |
| `calibration_quality_score` | whether metric scale/projection is trustworthy | dataset metadata |
| `projection_cost` | how much physical correction was needed | constraint layer |
