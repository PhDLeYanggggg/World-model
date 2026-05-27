# User Action Required: Stage42-EH Source Terms Confirmation Intake

No external source is conversion-ready yet. The agent must not fill legal acceptance fields for you.

Fill this file after official verification:

`outputs/stage42_long_research/source_terms_confirmation_intake_template_stage42.json`

## Priority Targets

### 1. ucy_crowd_original

- official_url_from_prior_audit: https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data
- domain: `UCY`
- estimated t50/t100 after terms: `9554` / `5605`
- source-CV after terms: `True`
- required fields: `terms_accepted_by_user, terms_acceptance_date, official_terms_url, accepted_terms_version_or_access_date, allowed_use, redistribution_allowed, derived_data_allowed, local_path, source_identity, confirmed_by_user`
- next validator command: `.venv-pytorch/bin/python run_stage42_source_terms_confirmation_validator.py`

### 2. eth_biwi_original

- official_url_from_prior_audit: https://vision.ee.ethz.ch/datsets.html
- domain: `ETH_UCY`
- estimated t50/t100 after terms: `506` / `91`
- source-CV after terms: `True`
- required fields: `terms_accepted_by_user, terms_acceptance_date, official_terms_url, accepted_terms_version_or_access_date, allowed_use, redistribution_allowed, derived_data_allowed, local_path, source_identity, confirmed_by_user`
- next validator command: `.venv-pytorch/bin/python run_stage42_source_terms_confirmation_validator.py`

### 3. aerialmpt_or_other_topdown

- official_url_from_prior_audit: user_or_web_verified_official_url_required
- domain: `other_topdown`
- estimated t50/t100 after terms: `0` / `0`
- source-CV after terms: `False`
- required fields: `terms_accepted_by_user, terms_acceptance_date, official_terms_url, accepted_terms_version_or_access_date, allowed_use, redistribution_allowed, derived_data_allowed, local_path, source_identity, confirmed_by_user`
- next validator command: `.venv-pytorch/bin/python run_stage42_source_terms_confirmation_validator.py`

### 4. opentraj_toolkit

- official_url_from_prior_audit: https://github.com/crowdbotp/OpenTraj
- domain: `OpenTraj`
- estimated t50/t100 after terms: `0` / `0`
- source-CV after terms: `False`
- required fields: `terms_accepted_by_user, terms_acceptance_date, official_terms_url, accepted_terms_version_or_access_date, allowed_use, redistribution_allowed, derived_data_allowed, local_path, source_identity, confirmed_by_user`
- next validator command: `.venv-pytorch/bin/python run_stage42_source_terms_confirmation_validator.py`

### 5. trajnetplusplus_official

- official_url_from_prior_audit: https://www.epfl.ch/schools/enac/about/open-science/open-research/trajnet/
- domain: `TrajNet`
- estimated t50/t100 after terms: `0` / `0`
- source-CV after terms: `False`
- required fields: `terms_accepted_by_user, terms_acceptance_date, official_terms_url, accepted_terms_version_or_access_date, allowed_use, redistribution_allowed, derived_data_allowed, local_path, source_identity, confirmed_by_user`
- next validator command: `.venv-pytorch/bin/python run_stage42_source_terms_confirmation_validator.py`

## Safety Rule

Do not download, convert, evaluate, or claim metric/seconds-level evidence from these sources until a later validator + no-leakage + source-CV stage passes.
