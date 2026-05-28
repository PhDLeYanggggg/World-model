# Stage42-JO Local Calibrated Source Guarded Conversion Preflight

- source: `fresh_stage42_jo_local_calibrated_source_guarded_conversion_preflight`
- generated_at_utc: `2026-05-28T21:51:52.559975+00:00`
- git_commit: `f090537`
- input_report: `outputs/stage42_long_research/local_calibrated_source_support_intake_stage42.json`
- input_hash: `1b77a8f790fb4165e235dec33261bc646e66145492dc473a38af427fbd159d59`
- gate: `13 / 13`
- verdict: `stage42_jo_local_calibrated_source_guarded_preflight_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-JO consumes Stage42-JN local support-candidate evidence and builds a guarded conversion preflight.
- This stage does not convert Town-Center, Wild-Track, or PETS into the benchmark because dataset-specific terms are not confirmed.
- Future endpoints may appear only as future labels after legal conversion; they are not inference inputs in this preflight.
- Local calibration files are treated as projection hints, not permission for global metric/seconds claims.
- Stage5C and SMC remain disabled.

## Summary

- decision: `guarded_conversion_preflight_blocked_pending_user_terms`
- technical_ready_after_terms: `['Town-Center', 'Wild-Track', 'PETS-2009-S2L1']`
- conversion_allowed_now: `[]`
- blocked_by_terms: `['Town-Center', 'Wild-Track', 'PETS-2009-S2L1']`
- blocked_by_geometry_audit: `['Town-Center', 'Wild-Track', 'PETS-2009-S2L1']`
- next_action: User confirms dataset-specific official terms/source identity, then rerun this preflight before guarded conversion.

## Candidate Preflights

| dataset | technical ready after terms | conversion now | t50 | t100 | legal blockers | geometry blockers | status |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| `Town-Center` | `True` | `False` | 60417 | 50132 | `['terms_not_accepted_by_user', 'source_identity_not_confirmed', 'conversion_scope_not_confirmed', 'official_terms_url_missing', 'allowed_use_missing']` | `['world_projection_not_integrated', 'pixel_coordinate_requires_source_specific_projection_audit']` | `not_run_user_terms_required` |
| `Wild-Track` | `True` | `False` | 2539 | 1770 | `['terms_not_accepted_by_user', 'source_identity_not_confirmed', 'conversion_scope_not_confirmed', 'official_terms_url_missing', 'allowed_use_missing']` | `['ground_grid_requires_dataset_specific_geometry_audit_before_metric_claim']` | `not_run_user_terms_required` |
| `PETS-2009-S2L1` | `True` | `False` | 3700 | 2768 | `['terms_not_accepted_by_user', 'source_identity_not_confirmed', 'conversion_scope_not_confirmed', 'official_terms_url_missing', 'allowed_use_missing']` | `['world_projection_not_integrated', 'pixel_coordinate_requires_source_specific_projection_audit']` | `not_run_user_terms_required` |

## Conversion Contract

- Do not convert any candidate until user-confirmed official terms/source identity and conversion scope are recorded.
- After confirmation, rerun this preflight and then run a guarded converter, no-leakage audit, source-CV split, strongest baseline, and protected policy evaluation.
- Future endpoint coordinates may be materialized only as supervised labels/evaluation labels, never as inference inputs.
- Source-specific calibration can support restricted geometry audits only after conversion; it does not authorize global metric or seconds-level claims.

## No-Leakage And Claim Boundary

- no_leakage: `{'future_endpoint_input': False, 'central_velocity': False, 'test_endpoint_goals': False, 'test_threshold_tuning': False, 'conversion_executed': False, 'preflight_only': True}`
- claim_boundary: `{'true_3d': False, 'foundation_world_model': False, 'global_metric_or_seconds_claim': False, 'converted_external_support_source': False, 'stage5c_executed': False, 'smc_enabled': False}`
