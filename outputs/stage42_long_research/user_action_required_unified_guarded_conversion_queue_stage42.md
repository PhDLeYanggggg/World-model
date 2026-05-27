# Stage42-FT User Action Required: Unified Guarded Conversion Queue

The unified guarded conversion queue is empty. This is correct while source terms/path/source identity are not confirmed.

## Required Files To Fill Manually

- global source intake: `outputs/stage42_long_research/source_terms_confirmation_intake_template_stage42.json`
- UCY H100 candidate intake: `outputs/stage42_long_research/ucy_h100_candidate_terms_template_stage42.json`

## Blocked Items

| scope | dataset | candidate | blockers |
| --- | --- | --- | --- |
| `global_source_manifest` | `ucy_crowd_original` | `ucy_crowd_original` | manual_terms_or_application_required, no_independent_t50_candidate, terms_not_accepted, terms_acceptance_date_missing, allowed_use_missing, local_path_confirmation_missing, source_identity_missing |
| `global_source_manifest` | `eth_biwi_original` | `eth_biwi_original` | manual_terms_or_application_required, no_independent_t50_candidate, terms_not_accepted, terms_acceptance_date_missing, allowed_use_missing, local_path_confirmation_missing, source_identity_missing |
| `global_source_manifest` | `trajnetplusplus_official` | `trajnetplusplus_official` | manual_terms_or_application_required, no_independent_t50_candidate, terms_not_accepted, terms_acceptance_date_missing, allowed_use_missing, local_path_confirmation_missing, source_identity_missing |
| `global_source_manifest` | `opentraj_toolkit` | `opentraj_toolkit` | no_independent_t50_candidate, terms_not_accepted, terms_acceptance_date_missing, allowed_use_missing, local_path_confirmation_missing, source_identity_missing |
| `global_source_manifest` | `aerialmpt_or_other_topdown` | `aerialmpt_or_other_topdown` | local_path_missing, schema_not_parseable, manual_terms_or_application_required, no_independent_t50_candidate, terms_not_accepted, terms_acceptance_date_missing, allowed_use_missing, local_path_confirmation_missing, source_identity_missing |
| `ucy_h100_candidate` | `ucy_crowd_original` | `UCY_zara02::obsmat` | terms_not_accepted, terms_acceptance_date_missing, allowed_use_missing, redistribution_policy_unknown, derived_data_policy_unknown, local_path_confirmation_missing, source_identity_missing, confirmed_by_user_missing |
| `ucy_h100_candidate` | `ucy_crowd_original` | `UCY_zara01::obsmat` | terms_not_accepted, terms_acceptance_date_missing, allowed_use_missing, redistribution_policy_unknown, derived_data_policy_unknown, local_path_confirmation_missing, source_identity_missing, confirmed_by_user_missing |
| `ucy_h100_candidate` | `ucy_crowd_original` | `UCY_students03::obsmat_px` | terms_not_accepted, terms_acceptance_date_missing, allowed_use_missing, redistribution_policy_unknown, derived_data_policy_unknown, local_path_confirmation_missing, source_identity_missing, confirmed_by_user_missing |
| `ucy_h100_candidate` | `ucy_crowd_original` | `UCY_students03::obsmat` | terms_not_accepted, terms_acceptance_date_missing, allowed_use_missing, redistribution_policy_unknown, derived_data_policy_unknown, local_path_confirmation_missing, source_identity_missing, confirmed_by_user_missing |
| `ucy_h100_candidate` | `ucy_crowd_original` | `UCY_students01::students001` | terms_not_accepted, terms_acceptance_date_missing, allowed_use_missing, redistribution_policy_unknown, derived_data_policy_unknown, local_path_confirmation_missing, source_identity_missing, confirmed_by_user_missing |
| `ucy_h100_candidate` | `ucy_crowd_original` | `UCY_students03::students003` | terms_not_accepted, terms_acceptance_date_missing, allowed_use_missing, redistribution_policy_unknown, derived_data_policy_unknown, local_path_confirmation_missing, source_identity_missing, confirmed_by_user_missing |
