# Stage42-FU Module Contribution Ledger

- source: `fresh_stage42_module_contribution_ledger_from_aa_y_bw_ec_dp_de`
- generated_at_utc: `2026-05-27T10:12:19.963927+00:00`
- gate: `14 / 14`
- verdict: `stage42_fu_module_contribution_ledger_pass`
- modules_total: `11`
- supported_or_necessary_modules: `6`
- main_claim_allowed_modules: `['history', 'domain_expert', 'safe_switch', 'teacher_floor', 'group_consistency_full_waypoint', 'full_waypoint_shape', 'endpoint_bridge']`
- blocked_or_auxiliary_modules: `['scene_goal', 'neighbor_interaction', 'JEPA', 'Transformer']`

## Module Ledger

| module | status | source | main claim | claim | limitation |
| --- | --- | --- | ---: | --- | --- |
| `history` | `supported_main_claim` | `fresh_run` | True | History tokens are a strong positive contributor in the retrained causal sequence ablation. | Still dataset-local raw-frame 2.5D; not metric/seconds evidence. |
| `domain_expert` | `supported_main_claim` | `fresh_run` | True | Domain expert routing contributes positively in retrained source-level evidence. | Does not prove foundation-style domain generalization; source support remains bounded. |
| `safe_switch` | `supported_safety_mechanism` | `fresh_run` | True | Safe switch/fallback is necessary for deployability; no-safe-switch harms at least one key slice. | Safe switch is a protected deployment mechanism, not an ungated neural dynamics claim. |
| `teacher_floor` | `necessary_not_removable` | `fresh_stage42_bw_safety_floor_necessity_audit` | True | Teacher/Stage37 floor remains required: ungated neural variants are not deployable due easy degradation. | This means current best model is protected, not floor-free neural world dynamics. |
| `group_consistency_full_waypoint` | `supported_source_level_claim` | `fresh_synthesis_from_stage42_dy_dz_ea_dp` | True | Explicit group-consistency full-waypoint dynamics has source-level bootstrap-backed positive-safe evidence. | Not an ungated/global primary full-waypoint replacement. |
| `full_waypoint_shape` | `partial_horizon_shape_support` | `fresh_run` | True | Full-waypoint shape is useful as protected horizon/shape evidence, especially t50/t100 raw-frame slices. | It does not replace endpoint-linear/teacher floor on all and hard/failure. |
| `endpoint_bridge` | `supported_floor_component` | `fresh_run` | True | Endpoint-linear bridge remains an important safety/accuracy floor component. | Endpoint success alone cannot be claimed as learned full-waypoint dynamics. |
| `scene_goal` | `weak_or_mixed_not_main_claim` | `fresh_run` | False | Scene/goal has weak retrained evidence but not enough to be an independent main contribution under current protocols. | Current context closure and prior goal/scene expert runs do not support a main independent contribution claim. |
| `neighbor_interaction` | `weak_or_mixed_not_main_claim` | `fresh_run` | False | Neighbor/interaction has small or mixed contribution; current graph residual protocol is negative. | Current graph/interaction rows remain below baseline-family control. |
| `JEPA` | `blocked_negative_or_inconclusive` | `cached_verified` | False | JEPA cannot be claimed as a downstream contributor in current evidence. | Existing JEPA is non-collapse but downstream lift is not stable. |
| `Transformer` | `fresh_proxy_negative_or_inconclusive` | `fresh_run` | False | Transformer contribution is not independently proven as a main claim under the current proxy. | This is a proxy boundary, not a complete no-Transformer retrain claim. |

## Paper Claim Boundary

- paper_claim_core: `['history', 'domain_expert', 'safe_switch', 'teacher_floor', 'group_consistency_full_waypoint']`
- paper_claim_blocked: `['JEPA_downstream_lift', 'ungated_neural_dynamics', 'scene_goal_independent_main_claim', 'neighbor_interaction_independent_main_claim', 'global_metric_seconds_claim']`
- Claims remain protected dataset-local/raw-frame 2.5D; no metric/seconds, no true 3D, no Stage5C, no SMC.
