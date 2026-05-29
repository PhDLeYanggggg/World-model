# Stage42-JQ Local Calibrated Source Terms Validation

- source: `fresh_stage42_jq_local_calibrated_source_terms_validator`
- generated_at_utc: `2026-05-29T01:35:41.758792+00:00`
- git_commit: `ebaa09f`
- input_hash: `db9ce5c22235ba4032e586fcc431f82547d3c228ce1608d3642764ed0ba554b6`
- gate: `14 / 14`
- verdict: `stage42_jq_local_calibrated_source_terms_validation_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-JQ validates user-filled local calibrated source terms; it does not accept terms for the user.
- Blank or partially filled terms templates block conversion.
- Conversion readiness requires explicit user terms acceptance, official/source URL confirmation, allowed use, source identity, scope confirmation, and local path existence.
- No download, conversion, training, evaluation, metric/seconds claim, Stage5C, or SMC is executed here.

## Summary

- decision: `blocked_until_user_fills_terms_template`
- datasets_validated: `3`
- terms_accepted_rows: `0`
- conversion_ready_rows: `0`
- conversion_allowed_now_count: `0`
- converted_now: `0`
- evaluated_now: `0`
- blocked_rows: `['Town-Center', 'Wild-Track', 'PETS-2009-S2L1']`
- ready_for_future_guarded_conversion: `[]`

## Validation Table

| dataset | accepted | ready | blockers | warnings | t50 | t100 |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| `Town-Center` | `False` | `False` | `['official_url_missing', 'official_terms_url_missing', 'license_name_missing', 'terms_not_accepted_by_user', 'accepted_by_user_missing', 'accepted_at_utc_missing', 'allowed_use_missing', 'source_identity_confirmed_false', 'conversion_scope_confirmed_false', 'official_url_not_confirmed_against_prefill']` | `['commercial_use_allowed_not_recorded', 'derived_data_allowed_not_recorded', 'redistribution_allowed_not_recorded', 'low_source_confidence_requires_extra_manual_review']` | `60417` | `50132` |
| `Wild-Track` | `False` | `False` | `['official_url_missing', 'official_terms_url_missing', 'license_name_missing', 'terms_not_accepted_by_user', 'accepted_by_user_missing', 'accepted_at_utc_missing', 'allowed_use_missing', 'source_identity_confirmed_false', 'conversion_scope_confirmed_false', 'official_url_not_confirmed_against_prefill']` | `['commercial_use_allowed_not_recorded', 'derived_data_allowed_not_recorded', 'redistribution_allowed_not_recorded']` | `2539` | `1770` |
| `PETS-2009-S2L1` | `False` | `False` | `['official_url_missing', 'official_terms_url_missing', 'license_name_missing', 'terms_not_accepted_by_user', 'accepted_by_user_missing', 'accepted_at_utc_missing', 'allowed_use_missing', 'source_identity_confirmed_false', 'conversion_scope_confirmed_false', 'official_url_not_confirmed_against_prefill']` | `['commercial_use_allowed_not_recorded', 'derived_data_allowed_not_recorded', 'redistribution_allowed_not_recorded']` | `3700` | `2768` |

## Interpretation

- This stage is an intake validator only. It does not convert or evaluate any source.
- A row marked `conversion_ready` would only be eligible for a later guarded conversion stage; it is still not converted here.
- Blank templates correctly remain blocked.
- Dataset-local calibration hints are not global metric or seconds-level claims.
