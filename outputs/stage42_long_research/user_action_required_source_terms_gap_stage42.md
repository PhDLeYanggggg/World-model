# User Action Required: Stage42-EF Source Terms Gap

No source is conversion-ready. Fill `outputs/stage42_long_research/source_terms_confirmation_template_stage42.json` only after official terms/path/source verification.

## Priority Order

### 1. ucy_crowd_original

- official_url: https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data
- domain: `UCY`
- missing fields: terms_accepted_by_user, terms_acceptance_date, allowed_use, local_path, source_identity
- estimated t50/t100 windows after terms: `9554` / `5605`
- source-CV after terms: `True`
- next command after confirmation: `.venv-pytorch/bin/python run_stage42_source_terms_confirmation_validator.py`
- then required future stage: guarded source conversion + no-leakage + source-CV evaluation; Stage42-EF does not convert

### 2. eth_biwi_original

- official_url: https://vision.ee.ethz.ch/datsets.html
- domain: `ETH_UCY`
- missing fields: terms_accepted_by_user, terms_acceptance_date, allowed_use, local_path, source_identity
- estimated t50/t100 windows after terms: `506` / `91`
- source-CV after terms: `True`
- next command after confirmation: `.venv-pytorch/bin/python run_stage42_source_terms_confirmation_validator.py`
- then required future stage: guarded source conversion + no-leakage + source-CV evaluation; Stage42-EF does not convert

### 3. aerialmpt_or_other_topdown

- official_url: user_or_web_verified_official_url_required
- domain: `other_topdown`
- missing fields: terms_accepted_by_user, terms_acceptance_date, allowed_use, local_path, source_identity
- estimated t50/t100 windows after terms: `0` / `0`
- source-CV after terms: `False`
- next command after confirmation: `.venv-pytorch/bin/python run_stage42_source_terms_confirmation_validator.py`
- then required future stage: guarded source conversion + no-leakage + source-CV evaluation; Stage42-EF does not convert

### 4. opentraj_toolkit

- official_url: https://github.com/crowdbotp/OpenTraj
- domain: `OpenTraj`
- missing fields: terms_accepted_by_user, terms_acceptance_date, allowed_use, local_path, source_identity
- estimated t50/t100 windows after terms: `0` / `0`
- source-CV after terms: `False`
- next command after confirmation: `.venv-pytorch/bin/python run_stage42_source_terms_confirmation_validator.py`
- then required future stage: guarded source conversion + no-leakage + source-CV evaluation; Stage42-EF does not convert

### 5. trajnetplusplus_official

- official_url: https://www.epfl.ch/schools/enac/about/open-science/open-research/trajnet/
- domain: `TrajNet`
- missing fields: terms_accepted_by_user, terms_acceptance_date, allowed_use, local_path, source_identity
- estimated t50/t100 windows after terms: `0` / `0`
- source-CV after terms: `False`
- next command after confirmation: `.venv-pytorch/bin/python run_stage42_source_terms_confirmation_validator.py`
- then required future stage: guarded source conversion + no-leakage + source-CV evaluation; Stage42-EF does not convert

Do not convert or evaluate until the validator reports conversion-ready targets and a later no-leakage/source-CV conversion stage passes.
