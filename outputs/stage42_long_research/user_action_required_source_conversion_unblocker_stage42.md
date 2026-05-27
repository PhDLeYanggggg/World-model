# User Action Required: Stage42-ED Source Conversion Unblocker

Stage42-ED did not download, convert, train, or evaluate data. The next step requires explicit user confirmation of official terms and source identity.

| priority | dataset | official URL | purpose | missing action |
| ---: | --- | --- | --- | --- |
| 1 | `ucy_crowd_original` | https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data | source-specific metric/time and source-CV repair | additional_t100_sources_needed=1, allowed_use_missing, legal_terms_blocked_targets=ucy_crowd_original, local_path_confirmation_missing, manual_terms_or_application_required, no_independent_t50_candidate, source_identity_missing, source_terms_confirmation_or_conversion_readiness_missing, terms_acceptance_date_missing, terms_not_accepted, train_only_t100_source_cv_support_missing |
| 2 | `eth_biwi_original` | https://vision.ee.ethz.ch/datsets.html | source-specific metric/time and source-CV repair | additional_t100_sources_needed=2, allowed_use_missing, legal_terms_blocked_targets=eth_biwi_original, local_path_confirmation_missing, manual_terms_or_application_required, no_independent_t50_candidate, source_identity_missing, source_terms_confirmation_or_conversion_readiness_missing, terms_acceptance_date_missing, terms_not_accepted, train_only_t100_source_cv_support_missing |
| 3 | `aerialmpt_or_other_topdown` | user_or_web_verified_official_url_required | source-diversity acquisition or identity repair | allowed_use_missing, local_path_confirmation_missing, local_path_missing, manual_terms_or_application_required, no_independent_t50_candidate, schema_not_parseable, source_identity_missing, terms_acceptance_date_missing, terms_not_accepted |
| 4 | `opentraj_toolkit` | https://github.com/crowdbotp/OpenTraj | source-diversity acquisition or identity repair | allowed_use_missing, local_path_confirmation_missing, no_independent_t50_candidate, source_identity_missing, terms_acceptance_date_missing, terms_not_accepted |
| 5 | `trajnetplusplus_official` | https://www.epfl.ch/schools/enac/about/open-science/open-research/trajnet/ | source-diversity acquisition or identity repair | additional_t100_sources_needed=1, allowed_use_missing, legal_terms_blocked_targets=trajnetplusplus_official, local_path_confirmation_missing, manual_terms_or_application_required, no_independent_t50_candidate, source_identity_missing, source_specific_metric_time_calibration_missing, source_terms_confirmation_or_conversion_readiness_missing, terms_acceptance_date_missing, terms_not_accepted, train_only_t100_source_cv_support_missing |

Required confirmation fields per dataset:

- `dataset_id`
- `official_url`
- `terms_accepted_by_user: true`
- `terms_acceptance_date`
- `allowed_use`
- `local_path`
- `source_identity`
- `notes`

After filling the confirmation template, run:

```bash
.venv-pytorch/bin/python run_stage42_source_terms_confirmation_validator.py
```
