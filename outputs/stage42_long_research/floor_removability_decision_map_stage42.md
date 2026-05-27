# Stage42-EN Floor Removability Decision Map

- source: `fresh_stage42_floor_removability_decision_map`
- generated_at_utc: `2026-05-27T02:55:35.905361+00:00`
- git_commit: `c740c24`
- input_hash: `352ee9ba2b1ed531816a6d61db9990e91f32bebdfc92df138b30ac33c6fb61e7`
- gate: `13 / 13`
- verdict: `stage42_en_floor_removability_decision_map_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-EN is a fresh synthesis audit over cached-verified safety-floor evidence; it does not train or tune a model.
- This audit distinguishes deployment fallback, teacher/floor rollout context, proximity guard, and narrow t50 fallback relaxation.
- future endpoints / waypoints remain supervised/evaluation labels only, never inference inputs.
- No central velocity, no test endpoints for goals, no test-threshold tuning.
- t+50 / t+100 remain raw-frame horizons; no seconds-level claim.
- dataset-local/raw-frame coordinates are not global metric coordinates.
- Stage5C latent generative was not executed.
- SMC was not enabled.

## Summary

- components_audited: `7`
- floor_free_neural_deployable: `False`
- safe_partial_floor_relaxation_available: `True`
- global_floor_removal_allowed: `False`
- teacher_floor_rollout_context_removal_allowed: `False`
- proximity_guard_required_for_safety_claim: `True`

## Decision Map

| component | decision | key metrics | deployment action |
| --- | --- | --- | --- |
| `ungated_neural_endpoint_or_full_waypoint` | `blocked` | ungated_endpoint_easy_degradation=124.59%, ungated_full_waypoint_easy_degradation=124.59%, easy_limit=2.00% | `do_not_deploy_ungated_neural` |
| `teacher_floor_rollout_context` | `required` | no_floor_rel_context_t50_delta=-9.21%, no_safe_baseline_context_t50_delta=-9.50% | `keep_teacher_floor_rollout_context` |
| `deployment_fallback_floor` | `required_globally_partial_relaxation_allowed` | floor_free_neural_deployable=False, repaired_t50_slices=['TrajNet|50', 'UCY|50'], global_t50_improvement_after_repair=28.97%, global_easy_degradation_after_repair=-37.05% | `allow_only_validation_backed_t50_slice_relaxation` |
| `proximity_guard` | `required_for_safety_sensitive_reporting` | no_guard_all_improvement=3.02%, no_guard_near_collision_delta=0.34%, guard_all_improvement=1.77%, guard_t50_improvement=1.07%, guard_near_collision_delta=-0.06% | `use_guarded_variant_for_safety_sensitive_claims` |
| `source_expansion_without_terms` | `blocked` | official_or_toolkit_source_candidates=4, conversion_ready_now=0, auto_download_allowed_now=0 | `wait_for_user_terms_path_source_identity_confirmation` |
| `t50_slice_relaxation::TrajNet|50` | `partial_supported` | rows=9198, t50_improvement=30.21%, hard_failure_improvement=30.21%, easy_degradation=-22.95%, switch_rate=95.26% | `slice_only_under_train_internal_validation_policy` |
| `t50_slice_relaxation::UCY|50` | `partial_supported` | rows=2340, t50_improvement=24.53%, hard_failure_improvement=24.53%, easy_degradation=-12.64%, switch_rate=65.00% | `slice_only_under_train_internal_validation_policy` |

## Interpretation

- Teacher/floor rollout context remains a core mechanism, not a removable implementation detail.
- Deployment fallback cannot be removed globally because ungated neural variants violate easy safety.
- Narrow t50 fallback relaxation is allowed only on validation-backed slices and still relies on teacher/floor context.
- The proximity guard is required for safety-sensitive claims because the no-guard variant has better ADE but worse near-collision.
- Source expansion remains blocked until the user confirms official terms, allowed use, local path, and source identity.

## Gate

| gate | pass |
| --- | ---: |
| `inputs_present` | True |
| `components_audited` | True |
| `ungated_neural_blocked` | True |
| `teacher_context_required` | True |
| `fallback_global_removal_blocked` | True |
| `partial_t50_relaxation_mapped` | True |
| `proximity_guard_required` | True |
| `source_terms_blocker_preserved` | True |
| `floor_free_neural_not_claimed` | True |
| `no_leakage_pass` | True |
| `no_metric_seconds_overclaim` | True |
| `stage5c_false` | True |
| `smc_false` | True |
