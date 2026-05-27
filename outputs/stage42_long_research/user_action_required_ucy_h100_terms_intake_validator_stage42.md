# Stage42-FS User Action Required: UCY H100 Terms Intake

No UCY H100 source is conversion-ready yet. Fill the candidate-level terms template only after manually verifying official UCY terms.

- official terms URL: https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data
- template to fill: `outputs/stage42_long_research/ucy_h100_candidate_terms_template_stage42.json`
- guarded queue output: `outputs/stage42_long_research/ucy_h100_guarded_conversion_queue_stage42.json`

## Blocked Candidates

| candidate | relative path | blockers |
| --- | --- | --- |
| `UCY_zara02::obsmat` | `UCY/zara02/obsmat.txt` | terms_not_accepted, terms_acceptance_date_missing, allowed_use_missing, redistribution_policy_unknown, derived_data_policy_unknown, local_path_confirmation_missing, source_identity_missing, confirmed_by_user_missing |
| `UCY_zara01::obsmat` | `UCY/zara01/obsmat.txt` | terms_not_accepted, terms_acceptance_date_missing, allowed_use_missing, redistribution_policy_unknown, derived_data_policy_unknown, local_path_confirmation_missing, source_identity_missing, confirmed_by_user_missing |
| `UCY_students03::obsmat_px` | `UCY/students03/obsmat_px.txt` | terms_not_accepted, terms_acceptance_date_missing, allowed_use_missing, redistribution_policy_unknown, derived_data_policy_unknown, local_path_confirmation_missing, source_identity_missing, confirmed_by_user_missing |
| `UCY_students03::obsmat` | `UCY/students03/obsmat.txt` | terms_not_accepted, terms_acceptance_date_missing, allowed_use_missing, redistribution_policy_unknown, derived_data_policy_unknown, local_path_confirmation_missing, source_identity_missing, confirmed_by_user_missing |
| `UCY_students01::students001` | `UCY/students01/students001.txt` | terms_not_accepted, terms_acceptance_date_missing, allowed_use_missing, redistribution_policy_unknown, derived_data_policy_unknown, local_path_confirmation_missing, source_identity_missing, confirmed_by_user_missing |
| `UCY_students03::students003` | `UCY/students03/students003.txt` | terms_not_accepted, terms_acceptance_date_missing, allowed_use_missing, redistribution_policy_unknown, derived_data_policy_unknown, local_path_confirmation_missing, source_identity_missing, confirmed_by_user_missing |

Do not count any row as converted until a later guarded conversion stage parses rows, rebuilds splits, passes no-leakage, and passes source-CV / h100 easy-safety CI.
