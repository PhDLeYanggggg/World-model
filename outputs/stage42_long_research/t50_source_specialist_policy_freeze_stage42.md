# Stage42-IM T50 Source-Specialist Policy Freeze

- source: `cached_verified_stage42_ik_il_t50_source_specialist_policy_freeze`
- generated_at_utc: `2026-05-28T01:59:58.125103+00:00`
- input_hash: `995a7bed2d1ad99b7a56be9a9dbe8a104d57de1ab5a7a8bf906a8ae58d5ea016`
- policy_hash: `9a73915ec3a74378def61d3a168f2d18c3fe0d6911fda1fcd971c8ada55ee1b2`
- policy_artifact: `outputs/stage42_long_research/frozen_t50_source_specialist_policy_stage42.json`
- gate: `22 / 22`
- verdict: `stage42_im_t50_source_specialist_policy_freeze_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-IM 冻结 Stage42-IK/IL 的 t50 source-specialist composition policy。
- IM 不训练新模型、不重新选择 threshold、不使用 test metrics 调参；只固化 source routing 与 claim boundary。
- future waypoints / endpoints 只作为 supervised/evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Frozen Routing Policy

- default_route: `stage42ii_t50_gain_harm_ensemble`
- ucy_route: `stage42x_row_aligned_ucy_full_waypoint_specialist`
- route_by: `['domain == UCY', 'source_file contains /UCY/zara03/crowds_zara03.txt']`
- new_threshold_selection: `False`
- uses_test_metrics_for_routing: `False`

## Compact Replay Summary

| metric | value |
| --- | ---: |
| ADE all | 0.158819 |
| ADE t50 | 0.104522 |
| ADE t50 CI low | 0.097328 |
| ADE t100 raw diagnostic | 0.180729 |
| ADE hard/failure | 0.163730 |
| ADE easy degradation | 0.000000 |
| FDE t50 | 0.263687 |
| switch rate | 0.306440 |

## Replay Checks

- metric_summary_exact_replay: `True`
- source_rows_exact_replay: `True`
- domain_rows_exact_replay: `True`
- il_delta_audit_exact_replay: `True`
- non_ucy_max_abs_delta: `0.000000101979`

## Interpretation

- Stage42-IM is the frozen deployment contract for the IK source-specialist composition.
- It does not train a new model and does not select new thresholds.
- It keeps Stage42-II as the default route and routes UCY to the row-aligned Stage42-X full-waypoint specialist.
- Claims remain protected dataset-local/raw-frame 2.5D only.
