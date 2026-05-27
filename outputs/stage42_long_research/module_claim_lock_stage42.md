# Stage42-GJ Module Claim Lock and Experiment Guard

- source: `fresh_stage42_gj_module_claim_lock_from_fu_z_dp_dq_gh`
- generated_at_utc: `2026-05-27T12:35:03.038327+00:00`
- git_commit: `34246ac`
- gate: `19 / 19`
- verdict: `stage42_gj_module_claim_lock_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-GJ 是 claim lock / experiment guard；它整合 FU/Z/DP/DQ/GH，不重新训练、不调 test threshold。
- future endpoints / waypoints 只允许作为 supervised/evaluation labels，不能作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Locked Claims

- paper_ready_scope: `protected_2p5d_raw_frame_world_state_candidate`
- not_ready_scope: `true_3d_metric_seconds_foundation_or_stage5c_smc`
- supported_main_modules_locked: `['history', 'domain_expert', 'safe_switch', 'teacher_floor', 'group_consistency_full_waypoint', 'full_waypoint_shape', 'endpoint_bridge']`
- blocked_main_modules_locked: `['scene_goal', 'neighbor_interaction', 'JEPA', 'Transformer']`
- context_protocol_status: `close_current_sequence_graph_residual_context_protocol`
- protected_full_waypoint_runtime_supported: `True`
- ungated_full_waypoint_deployable: `False`

## Calibrated Subset Boundary

- calibrated_subset_candidates_after_terms: `5`
- calibrated_subset_ready_now: `0`
- calibrated_t50_after_terms: `10060`
- calibrated_t100_after_terms: `5696`
- calibrated_domains_after_terms: `['ETH_UCY', 'UCY']`
- These are post-confirmation candidates only; no permission, conversion, training, or evaluation is claimed.

## Allowed Claims

- protected dataset-local/raw-frame 2.5D world-state candidate
- history/domain-expert/safe-switch/teacher-floor/group-consistency full-waypoint as supported modules
- protected source-level group-consistency full-waypoint runtime evidence
- post-confirmation calibrated subset candidate map as user-action plan only

## Forbidden Claims

- true 3D / foundation / global metric / seconds-level claim
- JEPA or Transformer as independent main contribution under current evidence
- scene/goal or neighbor/interaction as independent main contribution under current evidence
- ungated full-waypoint or ungated neural deployment
- post-confirmation candidates as converted/evaluated data before terms and guarded conversion
- Stage5C execution or SMC enablement

## Next Admissible Experiments

- terms-confirmed guarded conversion for calibrated ETH/UCY candidates, followed by no-leakage and source-CV
- changed-target context modeling only: gain/harm, switchability, or full-sequence objectives; do not repeat the closed residual sequence/graph protocol unchanged
- protected full-waypoint runtime replay or protocol-aligned evaluation; do not promote ungated full-waypoint
- source/horizon-specific h100 support repair after source/legal/calibration closure

## Gate

| gate | pass |
| --- | ---: |
| `fu_input_passed` | True |
| `z_input_passed` | True |
| `dp_input_passed` | True |
| `dq_input_passed` | True |
| `gh_input_passed` | True |
| `core_modules_locked` | True |
| `negative_modules_locked` | True |
| `context_residual_protocol_closed` | True |
| `protected_full_waypoint_supported` | True |
| `ungated_full_waypoint_blocked` | True |
| `calibrated_subset_candidates_recorded` | True |
| `calibrated_subset_not_claimed_ready` | True |
| `next_experiments_are_concrete` | True |
| `no_future_or_test_leakage` | True |
| `no_metric_seconds_overclaim` | True |
| `not_true3d_or_foundation` | True |
| `post_confirmation_candidates_not_overclaimed` | True |
| `stage5c_false` | True |
| `smc_false` | True |
