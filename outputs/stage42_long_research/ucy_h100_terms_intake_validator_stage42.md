# Stage42-FS UCY H100 Terms Intake Validator

- source: `fresh_stage42_ucy_h100_terms_intake_validator`
- generated_at_utc: `2026-05-27T09:54:34.661395+00:00`
- gate: `14 / 14`
- verdict: `stage42_fs_ucy_h100_terms_intake_validator_pass`
- input FR verdict: `stage42_fr_ucy_h100_terms_gated_preflight_pass`
- candidate rows validated: `6`
- target-family candidates: `2`
- terms-ready candidates: `0`
- guarded conversion queue count: `0`
- blocked candidates: `6`
- top blockers: `{'allowed_use_missing': 6, 'confirmed_by_user_missing': 6, 'derived_data_policy_unknown': 6, 'local_path_confirmation_missing': 6, 'redistribution_policy_unknown': 6, 'source_identity_missing': 6, 'terms_acceptance_date_missing': 6, 'terms_not_accepted': 6}`

## Validation Table

| candidate | source | target family | t100 windows | ready | candidate file | blockers |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `UCY_zara02::obsmat` | `UCY_zara02` | True | 2095 | False | `not_confirmed` | terms_not_accepted, terms_acceptance_date_missing, allowed_use_missing, redistribution_policy_unknown, derived_data_policy_unknown, local_path_confirmation_missing, source_identity_missing, confirmed_by_user_missing |
| `UCY_zara01::obsmat` | `UCY_zara01` | True | 97 | False | `not_confirmed` | terms_not_accepted, terms_acceptance_date_missing, allowed_use_missing, redistribution_policy_unknown, derived_data_policy_unknown, local_path_confirmation_missing, source_identity_missing, confirmed_by_user_missing |
| `UCY_students03::obsmat_px` | `UCY_students03` | False | 3415 | False | `not_confirmed` | terms_not_accepted, terms_acceptance_date_missing, allowed_use_missing, redistribution_policy_unknown, derived_data_policy_unknown, local_path_confirmation_missing, source_identity_missing, confirmed_by_user_missing |
| `UCY_students03::obsmat` | `UCY_students03` | False | 3413 | False | `not_confirmed` | terms_not_accepted, terms_acceptance_date_missing, allowed_use_missing, redistribution_policy_unknown, derived_data_policy_unknown, local_path_confirmation_missing, source_identity_missing, confirmed_by_user_missing |
| `UCY_students01::students001` | `UCY_students01` | False | 1949 | False | `not_confirmed` | terms_not_accepted, terms_acceptance_date_missing, allowed_use_missing, redistribution_policy_unknown, derived_data_policy_unknown, local_path_confirmation_missing, source_identity_missing, confirmed_by_user_missing |
| `UCY_students03::students003` | `UCY_students03` | False | 879 | False | `not_confirmed` | terms_not_accepted, terms_acceptance_date_missing, allowed_use_missing, redistribution_policy_unknown, derived_data_policy_unknown, local_path_confirmation_missing, source_identity_missing, confirmed_by_user_missing |

## Interpretation

- FS validates the FR candidate-level terms template and writes a guarded conversion queue.
- With the current blank/unconfirmed intake, the queue is empty and all candidates remain blocked.
- This is still not conversion, training, evaluation, metric/seconds evidence, Stage5C, or SMC.
