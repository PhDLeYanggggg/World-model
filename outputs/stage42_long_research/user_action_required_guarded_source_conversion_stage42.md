# User Action Required: Stage42-EJ Guarded Source Conversion

Stage42-EJ did not download, convert, train, or evaluate data. Conversion is blocked until the validator reports at least one conversion-ready target.

| dataset | official URL | missing confirmation | source-CV blockers |
| --- | --- | --- | --- |
| `ucy_crowd_original` | https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data | terms_not_accepted, terms_acceptance_date_missing, allowed_use_missing, local_path_confirmation_missing, source_identity_missing | manual_terms_or_application_required, no_independent_t50_candidate |
| `eth_biwi_original` | https://vision.ee.ethz.ch/datsets.html | terms_not_accepted, terms_acceptance_date_missing, allowed_use_missing, local_path_confirmation_missing, source_identity_missing | manual_terms_or_application_required, no_independent_t50_candidate |
| `trajnetplusplus_official` | https://www.epfl.ch/schools/enac/about/open-science/open-research/trajnet/ | terms_not_accepted, terms_acceptance_date_missing, allowed_use_missing, local_path_confirmation_missing, source_identity_missing | manual_terms_or_application_required, no_independent_t50_candidate |
| `opentraj_toolkit` | https://github.com/crowdbotp/OpenTraj | terms_not_accepted, terms_acceptance_date_missing, allowed_use_missing, local_path_confirmation_missing, source_identity_missing | no_independent_t50_candidate |
| `aerialmpt_or_other_topdown` | user_or_web_verified_official_url_required | terms_not_accepted, terms_acceptance_date_missing, allowed_use_missing, local_path_confirmation_missing, source_identity_missing | local_path_missing, schema_not_parseable, manual_terms_or_application_required, no_independent_t50_candidate |

Next safe steps:

1. Fill `outputs/stage42_long_research/source_terms_confirmation_intake_template_stage42.json` manually after checking official terms and local source identity.
2. Run `.venv-pytorch/bin/python run_stage42_source_terms_confirmation_validator.py`.
3. Rerun `.venv-pytorch/bin/python run_stage42_guarded_source_conversion_launcher.py`.
4. Only a later guarded converter may execute parser/no-leakage/source-CV work, and only for queued ready targets.
