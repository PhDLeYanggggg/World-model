# User Action Required: Stage43 Blocked Source Terms Validation

- template: `outputs/stage43_latent_state/stage43_blocked_source_terms_identity_template.json`
- validation_report: `outputs/stage43_latent_state/stage43_blocked_source_terms_validation.md`
- manifest: `outputs/stage43_latent_state/stage43_blocked_source_guarded_conversion_manifest.md`

Fill the template manually only after checking official source identity and terms. The agent must not fill acceptance fields for the user.

## Town-Center

- preferred_official_url_hint: `http://www.robots.ox.ac.uk/ActiveVision/Research/Projects/2009bbenfold_headpose/project.html`
- blockers: `['official_url_not_confirmed_by_user', 'official_terms_url_missing', 'license_name_missing', 'terms_not_accepted_by_user', 'accepted_by_user_missing', 'accepted_at_utc_missing', 'allowed_use_missing', 'source_identity_not_confirmed_by_user', 'calibration_projection_scope_not_confirmed_by_user', 'conversion_scope_not_confirmed_by_user', 'can_use_for_stage43_support_false']`
- warnings: `['source_confidence_requires_extra_review', 'metric_or_projection_not_verified_for_claims']`
- required: official URL confirmation, terms URL, license/use scope, accepted-by user/date, source identity, calibration projection scope, conversion scope, and Stage43 support permission.

## Wild-Track

- preferred_official_url_hint: `https://www.epfl.ch/labs/cvlab/data/data-wildtrack/`
- blockers: `['official_url_not_confirmed_by_user', 'official_terms_url_missing', 'license_name_missing', 'terms_not_accepted_by_user', 'accepted_by_user_missing', 'accepted_at_utc_missing', 'allowed_use_missing', 'source_identity_not_confirmed_by_user', 'calibration_projection_scope_not_confirmed_by_user', 'conversion_scope_not_confirmed_by_user', 'can_use_for_stage43_support_false']`
- warnings: `[]`
- required: official URL confirmation, terms URL, license/use scope, accepted-by user/date, source identity, calibration projection scope, conversion scope, and Stage43 support permission.

## PETS-2009-S2L1

- preferred_official_url_hint: `http://www.cvg.reading.ac.uk/PETS2009/a.html`
- blockers: `['official_url_not_confirmed_by_user', 'official_terms_url_missing', 'license_name_missing', 'terms_not_accepted_by_user', 'accepted_by_user_missing', 'accepted_at_utc_missing', 'allowed_use_missing', 'source_identity_not_confirmed_by_user', 'calibration_projection_scope_not_confirmed_by_user', 'conversion_scope_not_confirmed_by_user', 'can_use_for_stage43_support_false']`
- warnings: `['metric_or_projection_not_verified_for_claims']`
- required: official URL confirmation, terms URL, license/use scope, accepted-by user/date, source identity, calibration projection scope, conversion scope, and Stage43 support permission.

## TrajNet_biwi

- blockers: `['current_useful_biwi_support_entangled_with_heldout_test_source', 'heldout_source_disjoint_from_train_val_not_confirmed', 'independent_biwi_like_source_missing', 'new_independent_source_path_missing', 'official_url_not_confirmed', 'source_identity_not_confirmed', 'source_level_train_val_test_story_not_closed', 'terms_not_accepted']`
- required: an independent biwi-like source disjoint from held-out test source before any repair training.
