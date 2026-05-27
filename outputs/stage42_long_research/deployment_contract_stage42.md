# Stage42-EP Deployment Contract Guard

- source: `fresh_stage42_deployment_contract_guard`
- generated_at_utc: `2026-05-27T03:13:09.468067+00:00`
- git_commit: `a991b07`
- input_hash: `6b786e3fdf719526effde66368b0cd53c8fd64a60426ea28c9dd217146c64c98`
- gate: `16 / 16`
- verdict: `stage42_ep_deployment_contract_guard_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-EP turns deployment/paper claim boundaries into a machine-readable contract guard.
- This stage does not train, download, convert, or tune thresholds.
- future endpoints / waypoints 只允许作为 supervised/evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Contract Decisions

| request | allowed | status | role | denied reasons | required conditions |
| --- | ---: | --- | --- | --- | --- |
| `safety_sensitive_bridge_shape_deployment` | True | `allowed_protected` | `safety_sensitive_deployable_bridge_shape_policy` |  | use proximity_guard variant<br>keep Stage37/teacher floor<br>report dataset-local/raw-frame only<br>do not use future endpoint/waypoint inputs |
| `accuracy_priority_no_guard_reporting` | True | `allowed_diagnostic_only` | `accuracy_priority_diagnostic` |  | label as diagnostic/accuracy-priority<br>do not present as safety-sensitive deployable<br>include near-collision caveat |
| `source_level_full_waypoint_runtime` | True | `allowed_protocol_specific` | `source_level_full_waypoint_group_consistency_runtime_policy` |  | state source-level protocol baseline<br>keep protected floor context<br>do not rank-mix with endpoint-linear composer without baseline caveat<br>report raw-frame/dataset-local only |
| `global_floor_free_neural_deployment` | False | `blocked` | `forbidden` | global floor-free neural is not deployable<br>ungated endpoint/full-waypoint easy degradation violates the 2% safety limit |  |
| `teacher_floor_rollout_context_removal` | False | `blocked_required_mechanism` | `forbidden` | teacher/floor rollout context remains a core mechanism<br>removing floor/safe baseline rollout context hurts protected t50 |  |
| `validation_backed_t50_slice_relaxation` | True | `allowed_slice_only` | `partial_t50_floor_relaxation` |  | only on mapped t50 slices<br>use train/internal-validation policy<br>do not generalize to global floor-free deployment<br>keep teacher/floor rollout context |
| `source_conversion_without_user_terms` | False | `blocked_manual_terms_required` | `forbidden` | official links are not license acceptance<br>user must confirm terms, allowed use, local path, and source identity<br>auto download/conversion/evaluation are not allowed now |  |
| `metric_seconds_or_foundation_claim` | False | `forbidden` | `forbidden` | raw-frame/dataset-local evidence does not support metric or seconds-level claims<br>current model is not true 3D and not a foundation model<br>Stage5C and SMC remain disabled |  |
| `unknown_future_policy_request` | False | `unknown_request_blocked_by_default` | `forbidden` | unknown request is blocked until explicitly added to the contract |  |

## Deployment Defaults

- safety_sensitive_default: `proximity_guard`
- source_level_runtime_candidate: `group_consistency_full_waypoint_runtime`
- accuracy_priority_diagnostic: `no_proximity_guard`
- global_floor_free_neural: `blocked`
- source_conversion_without_terms: `blocked`
- metric_seconds_foundation_claim: `blocked`

## Gate

| gate | pass |
| --- | ---: |
| `dn_input_passed` | True |
| `em_input_passed` | True |
| `en_input_passed` | True |
| `eo_input_passed` | True |
| `safety_sensitive_default_allowed` | True |
| `no_guard_diagnostic_only` | True |
| `source_level_runtime_protocol_specific` | True |
| `floor_free_neural_blocked` | True |
| `teacher_context_removal_blocked` | True |
| `partial_t50_slice_relaxation_allowed` | True |
| `source_conversion_without_terms_blocked` | True |
| `metric_seconds_foundation_blocked` | True |
| `unknown_requests_default_deny` | True |
| `no_metric_seconds_overclaim` | True |
| `stage5c_false` | True |
| `smc_false` | True |
