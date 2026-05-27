# User Action Required: Stage42-GN Source Confirmation Priority Board

Fill only after checking the official dataset terms yourself. The agent must not accept terms for you.

Recommended order:

## 1. ucy_crowd_original (UCY)

- official_url: `https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data`
- value_class: `calibrated_t50_t100_unlock`
- post-confirmation t50/t100 opportunity: `9554 / 5605`
- calibrated t50/t100 opportunity: `9554 / 5605`
- suggested path to confirm, if it is truly the official/source dataset: `external_data/OpenTraj/datasets/UCY`
- missing fields: terms_accepted_by_user, terms_acceptance_date, accepted_terms_version_or_access_date, allowed_use, redistribution_allowed, derived_data_allowed, local_path, source_identity, confirmed_by_user
- after filling, run:
  `.venv-pytorch/bin/python run_stage42_source_terms_confirmation_validator.py`
  `.venv-pytorch/bin/python run_stage42_source_conversion_contract.py`
  `.venv-pytorch/bin/python run_stage42_guarded_conversion_harness.py`

## 2. eth_biwi_original (ETH_UCY)

- official_url: `https://vision.ee.ethz.ch/datsets.html`
- value_class: `calibrated_t50_t100_unlock`
- post-confirmation t50/t100 opportunity: `506 / 91`
- calibrated t50/t100 opportunity: `506 / 91`
- suggested path to confirm, if it is truly the official/source dataset: `external_data/OpenTraj/datasets/ETH`, `external_data/OpenTraj/datasets/ETH-Person`
- missing fields: terms_accepted_by_user, terms_acceptance_date, accepted_terms_version_or_access_date, allowed_use, redistribution_allowed, derived_data_allowed, local_path, source_identity, confirmed_by_user
- after filling, run:
  `.venv-pytorch/bin/python run_stage42_source_terms_confirmation_validator.py`
  `.venv-pytorch/bin/python run_stage42_source_conversion_contract.py`
  `.venv-pytorch/bin/python run_stage42_guarded_conversion_harness.py`

## 3. opentraj_toolkit (OpenTraj)

- official_url: `https://github.com/crowdbotp/OpenTraj`
- value_class: `low_or_diagnostic_unlock`
- post-confirmation t50/t100 opportunity: `0 / 0`
- calibrated t50/t100 opportunity: `0 / 0`
- suggested path to confirm, if it is truly the official/source dataset: `external_data/OpenTraj`
- missing fields: terms_accepted_by_user, terms_acceptance_date, accepted_terms_version_or_access_date, allowed_use, redistribution_allowed, derived_data_allowed, local_path, source_identity, confirmed_by_user
- after filling, run:
  `.venv-pytorch/bin/python run_stage42_source_terms_confirmation_validator.py`
  `.venv-pytorch/bin/python run_stage42_source_conversion_contract.py`
  `.venv-pytorch/bin/python run_stage42_guarded_conversion_harness.py`

## 4. trajnetplusplus_official (TrajNet)

- official_url: `https://www.epfl.ch/schools/enac/about/open-science/open-research/trajnet/`
- value_class: `low_or_diagnostic_unlock`
- post-confirmation t50/t100 opportunity: `0 / 0`
- calibrated t50/t100 opportunity: `0 / 0`
- suggested path to confirm, if it is truly the official/source dataset: `external_data/OpenTraj/datasets/TrajNet`, `external_data/OpenTraj/datasets/TrajNet++`
- missing fields: terms_accepted_by_user, terms_acceptance_date, accepted_terms_version_or_access_date, allowed_use, redistribution_allowed, derived_data_allowed, local_path, source_identity, confirmed_by_user
- after filling, run:
  `.venv-pytorch/bin/python run_stage42_source_terms_confirmation_validator.py`
  `.venv-pytorch/bin/python run_stage42_source_conversion_contract.py`
  `.venv-pytorch/bin/python run_stage42_guarded_conversion_harness.py`

## 5. aerialmpt_or_other_topdown (other_topdown)

- official_url: `user_or_web_verified_official_url_required`
- value_class: `low_or_diagnostic_unlock`
- post-confirmation t50/t100 opportunity: `0 / 0`
- calibrated t50/t100 opportunity: `0 / 0`
- suggested path to confirm, if it is truly the official/source dataset: `data/aerialmpt/DLR_AerialMPT_Dataset.zip`
- missing fields: terms_accepted_by_user, terms_acceptance_date, official_terms_url, accepted_terms_version_or_access_date, allowed_use, redistribution_allowed, derived_data_allowed, local_path, source_identity, confirmed_by_user
- after filling, run:
  `.venv-pytorch/bin/python run_stage42_source_terms_confirmation_validator.py`
  `.venv-pytorch/bin/python run_stage42_source_conversion_contract.py`
  `.venv-pytorch/bin/python run_stage42_guarded_conversion_harness.py`

Do not count this priority board as permission, conversion, feature store, no-leakage audit, source-CV, or model evidence.
