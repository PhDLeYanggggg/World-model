# Frozen Candidate Action-Class Ceiling

**Uses future labels. This is not an inference policy, learned improvement or independent confirmation.**

| Candidate condition | Binary oracle gain (%) | Whole-path scaling oracle gain (%) | Numerical optimistic gain (%) |
| --- | ---: | ---: | ---: |
| quality_control_row_uniform | 0.316084 | 0.364534 | 0.364534 |
| quality_control_scene_uniform | 0.365001 | 0.445148 | 0.445148 |
| quality_control_scene_track | 0.392458 | 0.482131 | 0.482131 |
| quality_control_scene_event_track | 1.014971 | 1.138160 | 1.138160 |
| directed_row_uniform | 0.317688 | 0.367275 | 0.367275 |
| directed_scene_uniform | 0.385717 | 0.470784 | 0.470784 |
| directed_scene_track | 0.398380 | 0.483540 | 0.483540 |
| directed_scene_event_track | 0.901972 | 1.001320 | 1.001320 |
| pooled_eight_candidates_per_seed | 1.626533 | 1.726175 | 1.726175 |

The pooled diagnostic may choose one of eight fixed candidates per row and seed, then one scalar in [0,1] for the entire path.
It does not bound arbitrary convex mixtures, per-waypoint correction, new predictors or world models.
Budget curves ignore joint constraints and use future outcomes. They are optimistic relaxations, not safety certificates.
