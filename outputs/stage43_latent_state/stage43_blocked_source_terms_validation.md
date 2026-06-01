# Stage43-BG Blocked Source Terms Validator

- source: `fresh_stage43_bg_blocked_source_terms_validator`
- result_source: `fresh_validation_of_stage43_bf_terms_identity_template`
- verdict: `stage43_bg_blocked_source_terms_validation_pass`
- gate: `13 / 13`
- datasets_validated: `3`
- ready_for_guarded_conversion_preflight_rows: `0`
- training_allowed_now: `0`

## Validation Table

| dataset | accepted | ready | t50 | t100 | blockers | warnings |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `Town-Center` | `False` | `False` | 60417 | 50132 | `['official_url_not_confirmed_by_user', 'official_terms_url_missing', 'license_name_missing', 'terms_not_accepted_by_user', 'accepted_by_user_missing', 'accepted_at_utc_missing', 'allowed_use_missing', 'source_identity_not_confirmed_by_user', 'calibration_projection_scope_not_confirmed_by_user', 'conversion_scope_not_confirmed_by_user', 'can_use_for_stage43_support_false']` | `['source_confidence_requires_extra_review', 'metric_or_projection_not_verified_for_claims']` |
| `Wild-Track` | `False` | `False` | 2539 | 1770 | `['official_url_not_confirmed_by_user', 'official_terms_url_missing', 'license_name_missing', 'terms_not_accepted_by_user', 'accepted_by_user_missing', 'accepted_at_utc_missing', 'allowed_use_missing', 'source_identity_not_confirmed_by_user', 'calibration_projection_scope_not_confirmed_by_user', 'conversion_scope_not_confirmed_by_user', 'can_use_for_stage43_support_false']` | `[]` |
| `PETS-2009-S2L1` | `False` | `False` | 3700 | 2768 | `['official_url_not_confirmed_by_user', 'official_terms_url_missing', 'license_name_missing', 'terms_not_accepted_by_user', 'accepted_by_user_missing', 'accepted_at_utc_missing', 'allowed_use_missing', 'source_identity_not_confirmed_by_user', 'calibration_projection_scope_not_confirmed_by_user', 'conversion_scope_not_confirmed_by_user', 'can_use_for_stage43_support_false']` | `['metric_or_projection_not_verified_for_claims']` |

## Biwi Independent Source

- ready_for_repair_training_preflight: `False`
- repair_training_allowed_now: `False`
- blockers: `['current_useful_biwi_support_entangled_with_heldout_test_source', 'heldout_source_disjoint_from_train_val_not_confirmed', 'independent_biwi_like_source_missing', 'new_independent_source_path_missing', 'official_url_not_confirmed', 'source_identity_not_confirmed', 'source_level_train_val_test_story_not_closed', 'terms_not_accepted']`

## Interpretation

The validator is doing the right thing: the Stage43-BF template is still blank, so every local candidate remains blocked. This keeps PETS, Town-Center, Wild-Track, and the biwi family out of conversion/training until source identity, terms, scope, and independent-source checks are actually closed.

## Claim Boundary

- Validator output is not permission.
- Manifest output is not conversion.
- No download, conversion, training, threshold tuning, or evaluation is executed.
- Dataset-local/raw-frame 2.5D only; no metric or seconds-level claim.
- Stage5C remains false and SMC remains false.
