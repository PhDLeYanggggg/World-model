# User Action Required: Stage42-GH Calibrated Subset Candidates

These rows are the highest-value source-specific metric/time candidates after legal/source confirmation.
Do not treat them as converted or evaluated data.

| rank | dataset | source | t50 | t100 | local evidence | required before use |
| ---: | --- | --- | ---: | ---: | --- | --- |
| 1 | `ucy_crowd_original` | `UCY_students03` | 6491 | 3413 | `source_specific_annotation_step_meter_coordinate_evidence` | terms_accepted_by_user, terms_acceptance_date, allowed_use, local_path, source_identity, confirmed_by_user, redistribution_allowed, derived_data_allowed |
| 2 | `ucy_crowd_original` | `UCY_zara02` | 2823 | 2095 | `source_specific_annotation_step_meter_coordinate_evidence` | terms_accepted_by_user, terms_acceptance_date, allowed_use, local_path, source_identity, confirmed_by_user, redistribution_allowed, derived_data_allowed |
| 3 | `ucy_crowd_original` | `UCY_zara01` | 240 | 97 | `source_specific_annotation_step_meter_coordinate_evidence` | terms_accepted_by_user, terms_acceptance_date, allowed_use, local_path, source_identity, confirmed_by_user, redistribution_allowed, derived_data_allowed |
| 4 | `eth_biwi_original` | `ETH_seq_eth` | 291 | 91 | `source_specific_annotation_step_meter_coordinate_evidence` | terms_accepted_by_user, terms_acceptance_date, allowed_use, local_path, source_identity, confirmed_by_user, redistribution_allowed, derived_data_allowed |
| 5 | `eth_biwi_original` | `ETH_seq_hotel` | 215 | 0 | `source_specific_annotation_step_meter_coordinate_evidence` | terms_accepted_by_user, terms_acceptance_date, allowed_use, local_path, source_identity, confirmed_by_user, redistribution_allowed, derived_data_allowed |

After filling official terms/path/source identity in `outputs/stage42_long_research/source_terms_confirmation_intake_template_stage42.json`, rerun:

```bash
.venv-pytorch/bin/python run_stage42_source_terms_confirmation_validator.py
.venv-pytorch/bin/python run_stage42_conversion_capability_intake_bridge.py
.venv-pytorch/bin/python run_stage42_post_confirmation_conversion_plan.py
.venv-pytorch/bin/python run_stage42_calibrated_post_confirmation_subset_plan.py
.venv-pytorch/bin/python run_stage42_unified_guarded_conversion_queue.py
```

Only a later guarded conversion/no-leakage/evaluation stage may produce restricted calibrated subset metrics.
