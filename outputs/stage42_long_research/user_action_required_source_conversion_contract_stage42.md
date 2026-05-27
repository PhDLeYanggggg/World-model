# User Action Required: Stage42-GL Source Conversion Contract

No source is conversion-ready yet. Fill the intake template only after checking official terms and confirming local source identity.

| dataset | official URL | missing fields | suggested raw paths to inspect |
| --- | --- | --- | --- |
| `ucy_crowd_original` | https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data | terms_accepted_by_user, terms_acceptance_date, accepted_terms_version_or_access_date, allowed_use, redistribution_allowed, derived_data_allowed, local_path, source_identity, confirmed_by_user | external_data/OpenTraj/datasets/UCY |
| `eth_biwi_original` | https://vision.ee.ethz.ch/datsets.html | terms_accepted_by_user, terms_acceptance_date, accepted_terms_version_or_access_date, allowed_use, redistribution_allowed, derived_data_allowed, local_path, source_identity, confirmed_by_user | external_data/OpenTraj/datasets/ETH, external_data/OpenTraj/datasets/ETH-Person |
| `aerialmpt_or_other_topdown` | user_or_web_verified_official_url_required | terms_accepted_by_user, terms_acceptance_date, official_terms_url, accepted_terms_version_or_access_date, allowed_use, redistribution_allowed, derived_data_allowed, local_path, source_identity, confirmed_by_user | data/aerialmpt/DLR_AerialMPT_Dataset.zip |
| `opentraj_toolkit` | https://github.com/crowdbotp/OpenTraj | terms_accepted_by_user, terms_acceptance_date, accepted_terms_version_or_access_date, allowed_use, redistribution_allowed, derived_data_allowed, local_path, source_identity, confirmed_by_user | external_data/OpenTraj |
| `trajnetplusplus_official` | https://www.epfl.ch/schools/enac/about/open-science/open-research/trajnet/ | terms_accepted_by_user, terms_acceptance_date, accepted_terms_version_or_access_date, allowed_use, redistribution_allowed, derived_data_allowed, local_path, source_identity, confirmed_by_user | external_data/OpenTraj/datasets/TrajNet, external_data/OpenTraj/datasets/TrajNet++ |

Allowed next commands after the user fills required fields:

1. `.venv-pytorch/bin/python run_stage42_source_terms_confirmation_validator.py`
2. `.venv-pytorch/bin/python run_stage42_guarded_source_conversion_launcher.py`
3. A future guarded converter may run only for queued targets and must redo no-leakage/source-CV/metric-time checks.
