# Stage42-FR UCY H100 Terms-Gated Conversion Preflight

- source: `fresh_stage42_ucy_h100_terms_gated_conversion_preflight`
- generated_at_utc: `2026-05-27T09:45:21.129143+00:00`
- gate: `14 / 14`
- verdict: `stage42_fr_ucy_h100_terms_gated_preflight_pass`
- input FQ verdict: `stage42_fq_h100_source_support_repair_queue_pass`
- dataset_id: `ucy_crowd_original`
- candidate rows: `6`
- target-family candidates: `2`
- conversion_preflight_ready_count: `0`
- blockers: `['terms_not_accepted', 'terms_acceptance_date_missing', 'allowed_use_missing', 'redistribution_policy_unknown', 'derived_data_policy_unknown', 'local_path_confirmation_missing', 'source_identity_missing', 'confirmed_by_user_missing']`
- template: `outputs/stage42_long_research/ucy_h100_candidate_terms_template_stage42.json`
- verification: `{'runner': '.venv-pytorch/bin/python run_stage42_ucy_h100_terms_gated_conversion_preflight.py -> 14/14', 'focused_pytest': '.venv-pytorch/bin/python -m pytest tests/test_stage42_ucy_h100_terms_gated_conversion_preflight.py -> 4 passed', 'full_pytest': '.venv-pytorch/bin/python -m pytest tests -> 840 passed'}`

## Candidate Table

| order | candidate_id | source_id | relative path | family | target match | max track | est h100 windows | ready | blockers |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | `UCY_zara02::obsmat` | `UCY_zara02` | `UCY/zara02/obsmat.txt` | `zara` | True | 583 | 2095 | False | terms_not_accepted, terms_acceptance_date_missing, allowed_use_missing, redistribution_policy_unknown, derived_data_policy_unknown, local_path_confirmation_missing, source_identity_missing, confirmed_by_user_missing |
| 2 | `UCY_zara01::obsmat` | `UCY_zara01` | `UCY/zara01/obsmat.txt` | `zara` | True | 197 | 97 | False | terms_not_accepted, terms_acceptance_date_missing, allowed_use_missing, redistribution_policy_unknown, derived_data_policy_unknown, local_path_confirmation_missing, source_identity_missing, confirmed_by_user_missing |
| 3 | `UCY_students03::obsmat_px` | `UCY_students03` | `UCY/students03/obsmat_px.txt` | `students` | False | 540 | 3415 | False | terms_not_accepted, terms_acceptance_date_missing, allowed_use_missing, redistribution_policy_unknown, derived_data_policy_unknown, local_path_confirmation_missing, source_identity_missing, confirmed_by_user_missing |
| 4 | `UCY_students03::obsmat` | `UCY_students03` | `UCY/students03/obsmat.txt` | `students` | False | 539 | 3413 | False | terms_not_accepted, terms_acceptance_date_missing, allowed_use_missing, redistribution_policy_unknown, derived_data_policy_unknown, local_path_confirmation_missing, source_identity_missing, confirmed_by_user_missing |
| 5 | `UCY_students01::students001` | `UCY_students01` | `UCY/students01/students001.txt` | `students` | False | 352 | 1949 | False | terms_not_accepted, terms_acceptance_date_missing, allowed_use_missing, redistribution_policy_unknown, derived_data_policy_unknown, local_path_confirmation_missing, source_identity_missing, confirmed_by_user_missing |
| 6 | `UCY_students03::students003` | `UCY_students03` | `UCY/students03/students003.txt` | `students` | False | 289 | 879 | False | terms_not_accepted, terms_acceptance_date_missing, allowed_use_missing, redistribution_policy_unknown, derived_data_policy_unknown, local_path_confirmation_missing, source_identity_missing, confirmed_by_user_missing |

## Interpretation

- FR is a file-level preflight, not a conversion stage.
- `UCY_zara02` is the highest-priority target-family h100 support candidate, but it is blocked until user-confirmed official terms/local path/source identity.
- `UCY_students03` has more estimated h100 windows but is not the zara target family for the current UCY|100 weak slice; it is secondary support.
- No raw data, cache, converted feature store, metric/seconds claim, Stage5C, or SMC is produced.
