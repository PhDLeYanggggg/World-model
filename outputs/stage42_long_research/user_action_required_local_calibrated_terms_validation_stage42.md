# User Action Required: Local Calibrated Terms Validation

- terms_template: `outputs/stage42_long_research/local_calibrated_source_terms_template_stage42.json`
- validation_report: `outputs/stage42_long_research/local_calibrated_source_terms_validation_stage42.md`

Fill the template manually only after checking official/source terms. The agent must not fill acceptance fields.

## Town-Center

- blockers: `['official_url_missing', 'official_terms_url_missing', 'license_name_missing', 'terms_not_accepted_by_user', 'accepted_by_user_missing', 'accepted_at_utc_missing', 'allowed_use_missing', 'source_identity_confirmed_false', 'conversion_scope_confirmed_false', 'official_url_not_confirmed_against_prefill']`
- warnings: `['commercial_use_allowed_not_recorded', 'derived_data_allowed_not_recorded', 'redistribution_allowed_not_recorded', 'low_source_confidence_requires_extra_manual_review']`
- local_path: `external_data/OpenTraj/datasets/Town-Center`
- official_url: ``
- required: official URL, official terms URL, license name, accepted-by user/date, allowed use, source identity, conversion scope.

## Wild-Track

- blockers: `['official_url_missing', 'official_terms_url_missing', 'license_name_missing', 'terms_not_accepted_by_user', 'accepted_by_user_missing', 'accepted_at_utc_missing', 'allowed_use_missing', 'source_identity_confirmed_false', 'conversion_scope_confirmed_false', 'official_url_not_confirmed_against_prefill']`
- warnings: `['commercial_use_allowed_not_recorded', 'derived_data_allowed_not_recorded', 'redistribution_allowed_not_recorded']`
- local_path: `external_data/OpenTraj/datasets/Wild-Track`
- official_url: ``
- required: official URL, official terms URL, license name, accepted-by user/date, allowed use, source identity, conversion scope.

## PETS-2009-S2L1

- blockers: `['official_url_missing', 'official_terms_url_missing', 'license_name_missing', 'terms_not_accepted_by_user', 'accepted_by_user_missing', 'accepted_at_utc_missing', 'allowed_use_missing', 'source_identity_confirmed_false', 'conversion_scope_confirmed_false', 'official_url_not_confirmed_against_prefill']`
- warnings: `['commercial_use_allowed_not_recorded', 'derived_data_allowed_not_recorded', 'redistribution_allowed_not_recorded']`
- local_path: `external_data/OpenTraj/datasets/PETS-2009/data`
- official_url: ``
- required: official URL, official terms URL, license name, accepted-by user/date, allowed use, source identity, conversion scope.
