# Stage42-EC Group-Consistency Contribution Audit

- source: `fresh_synthesis_from_stage42_dy_dz_ea_dp`
- generated_at_utc: `2026-05-27T01:32:32.411622+00:00`
- git_commit: `bf577ca`
- input_hash: `1e9ed979567750b2e73559b0fde3e01ea8ebcd071e8445f609cbbb46cf63aad5`
- gate: `17 / 17`
- verdict: `stage42_ec_group_consistency_contribution_audit_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-EC 是 contribution audit：fresh synthesis from DY/DZ/EA/DP，不重新训练，不调 threshold。
- future endpoints / waypoints 只作为 supervised/evaluation labels，不能作为 inference input。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Supported Contributions

| contribution | status | key evidence |
| --- | --- | --- |
| `explicit_group_consistency_full_waypoint` | `supported_source_level` | DY source-level promotion plus DZ dual-domain repair plus EA bootstrap. |
| `dual_domain_raw_frame_support` | `supported` | DZ reports UCY and TrajNet positive-safe domains; EA adds 2000-bootstrap positive-safe CIs for both domains. |
| `baseline_family_rollout_context` | `dominant_control_not_new_context_claim` | The baseline-family rollout context remains the dominant first-stage control; group consistency should be framed as a source-level physical consistency repair over this protected family. |

## Blocked Or Negative Contributions

| contribution | status | reason |
| --- | --- | --- |
| `scalar_loss_family_primary` | `blocked` | No scalar loss-family candidate beats Stage42-AM on the required all+hard promotion gate. |
| `current_sequence_graph_residual_context` | `closed_current_protocol` | Fresh Stage42-AR/AS reruns show that temporal sequence context and current-frame kNN graph context both reduce all/t50/hard-failure improvements relative to the baseline-family first-stage control. The dominant current signal remains baseline-family rollout context plus safety floor; the present residual context target is not extracting independent scene/goal/interaction value. |
| `goal_scene_main_claim` | `not_supported_under_current_protocols` | Current context closure and prior goal/scene expert runs do not support a main independent contribution claim. |
| `neighbor_interaction_main_claim` | `not_supported_under_current_protocols` | Current graph/interaction rows remain below baseline-family control. |
| `ungated_global_full_waypoint_replacement` | `blocked` | The supported group-consistency result is source-level protected evidence, not ungated or global primary replacement. |

## Statistical Evidence

- global all/t50/hard CI lows: `0.325616` / `0.265328` / `0.315115`.
- global easy degradation CI high: `-0.312813`.
- UCY positive-safe CI: `True`.
- TrajNet positive-safe CI: `True`.
- near@0.05 final-base delta high: `-0.006722`.

## Paper-Ready Claim

A paper can claim protected source-level group-consistency full-waypoint dynamics with UCY+TrajNet bootstrap-backed raw-frame evidence, while explicitly blocking scalar-loss, current sequence/graph residual context, ungated/global, metric/seconds, Stage5C, and SMC overclaims.

## Gate

| gate | pass |
| --- | ---: |
| `dy_input_passed` | True |
| `dz_input_passed` | True |
| `ea_input_passed` | True |
| `dp_input_passed` | True |
| `group_consistency_supported` | True |
| `dual_domain_bootstrap_supported` | True |
| `ucy_ci_positive_safe` | True |
| `trajnet_ci_positive_safe` | True |
| `near_collision_repaired` | True |
| `scalar_loss_family_blocked` | True |
| `context_residual_protocol_closed` | True |
| `goal_scene_overclaim_blocked` | True |
| `neighbor_interaction_overclaim_blocked` | True |
| `ungated_global_replacement_blocked` | True |
| `no_metric_seconds_overclaim` | True |
| `stage5c_false` | True |
| `smc_false` | True |
