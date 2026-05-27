# User Action Required: Stage42-FW Consolidated Source Actions

No source is newly conversion-ready in this stage. The rows below are the de-duplicated highest-impact actions required before any guarded external conversion/evaluation can be claimed.

Do not convert, evaluate, or claim metric/seconds/source-closed results until the relevant validator, guarded conversion, no-leakage, and source-CV stages pass.

## 1. FW-TERMS-ucy_crowd_original

- target: `ucy_crowd_original`
- domain: `UCY`
- category: `legal_terms_and_local_path`
- priority: `113`
- status: `not_run_user_action_required`
- official_url: https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data
- missing: `['terms_accepted_by_user', 'terms_acceptance_date', 'allowed_use', 'local_path', 'source_identity']`
- next user action: Confirm official terms and fill `ucy_crowd_original` fields in source_terms_confirmation_template_stage42.json: terms_accepted_by_user, terms_acceptance_date, allowed_use, local_path, source_identity.
- next commands after confirmation: `['.venv-pytorch/bin/python run_stage42_source_terms_confirmation_validator.py', '.venv-pytorch/bin/python run_stage42_unified_guarded_conversion_queue.py', '.venv-pytorch/bin/python run_stage42_guarded_source_conversion_launcher.py']`
- claim guard: Do not count this as converted/evaluated until validator, guarded conversion, no-leakage, and source-CV all pass.

## 2. FW-H100-TrajNet|100

- target: `TrajNet|100`
- domain: `TrajNet`
- category: `h100_weak_horizon_source_support`
- priority: `98`
- status: `not_run_user_action_required`
- official_url: not_recorded_in_consolidated_inputs
- missing: `['official longer TrajNet-compatible raw source', 'timing/geometry evidence', 'terms confirmation', 'local path']`
- next user action: Provide or legally confirm a longer official raw source for `TrajNet|100`; current local TrajNet snippets cannot support h100.
- next commands after confirmation: `['.venv-pytorch/bin/python run_stage42_unified_guarded_conversion_queue.py', '.venv-pytorch/bin/python run_stage42_guarded_source_conversion_launcher.py', '.venv-pytorch/bin/python run_stage42_h100_weak_horizon_source_support_audit.py']`
- claim guard: Candidate paths are inventory only; do not convert/evaluate or claim repair until legal terms, conversion, no-leakage, and source-CV pass.

## 3. FW-DOMAIN-TrajNet

- target: `TrajNet`
- domain: `TrajNet`
- category: `domain_closure`
- priority: `97`
- status: `not_run_open_blocker`
- official_url: not_recorded_in_consolidated_inputs
- missing: `['source_terms_confirmation_or_conversion_readiness_missing', 'train_only_t100_source_cv_support_missing', 'additional_t100_sources_needed=1', 'source_specific_metric_time_calibration_missing', 'legal_terms_blocked_targets=trajnetplusplus_official']`
- next user action: provide/confirm legal TrajNet++ or TrajNet-compatible long-track source with timing/geometry evidence, then rerun conversion, no-leakage, and train-only source-CV
- next commands after confirmation: `['.venv-pytorch/bin/python run_stage42_source_support_closure_audit.py']`
- claim guard: Domain remains not_closed until this action passes with no leakage and no terms blocker.

## 4. FW-DOMAIN-UCY

- target: `UCY`
- domain: `UCY`
- category: `domain_closure`
- priority: `97`
- status: `not_run_open_blocker`
- official_url: not_recorded_in_consolidated_inputs
- missing: `['source_terms_confirmation_or_conversion_readiness_missing', 'train_only_t100_source_cv_support_missing', 'additional_t100_sources_needed=1', 'legal_terms_blocked_targets=ucy_crowd_original']`
- next user action: confirm UCY original terms/source identity and add one independent t100-capable UCY source or source split before claiming stable t100
- next commands after confirmation: `['.venv-pytorch/bin/python run_stage42_source_support_closure_audit.py']`
- claim guard: Domain remains not_closed until this action passes with no leakage and no terms blocker.

## 5. FW-H100-UCY|100

- target: `UCY|100`
- domain: `UCY`
- category: `h100_weak_horizon_source_support`
- priority: `94`
- status: `not_run_user_action_required`
- official_url: not_recorded_in_consolidated_inputs
- missing: `['terms/license confirmation', 'guarded conversion', 'no-leakage audit', 'train-only source-CV']`
- next user action: Confirm terms/license for the listed local candidates for `UCY|100`, then run guarded conversion and source-CV.
- next commands after confirmation: `['.venv-pytorch/bin/python run_stage42_unified_guarded_conversion_queue.py', '.venv-pytorch/bin/python run_stage42_guarded_source_conversion_launcher.py', '.venv-pytorch/bin/python run_stage42_h100_weak_horizon_source_support_audit.py']`
- claim guard: Candidate paths are inventory only; do not convert/evaluate or claim repair until legal terms, conversion, no-leakage, and source-CV pass.

## 6. FW-DOMAIN-ETH_UCY

- target: `ETH_UCY`
- domain: `ETH_UCY`
- category: `domain_closure`
- priority: `90`
- status: `not_run_open_blocker`
- official_url: not_recorded_in_consolidated_inputs
- missing: `['source_terms_confirmation_or_conversion_readiness_missing', 'train_only_t100_source_cv_support_missing', 'additional_t100_sources_needed=2', 'legal_terms_blocked_targets=eth_biwi_original']`
- next user action: confirm ETH/BIWI or ETH-Person source terms and add enough independent t100-capable ETH_UCY train sources, then rerun source-CV without test tuning
- next commands after confirmation: `['.venv-pytorch/bin/python run_stage42_source_support_closure_audit.py']`
- claim guard: Domain remains not_closed until this action passes with no leakage and no terms blocker.

## 7. FW-TERMS-trajnetplusplus_official

- target: `trajnetplusplus_official`
- domain: `TrajNet`
- category: `legal_terms_and_local_path`
- priority: `60`
- status: `not_run_user_action_required`
- official_url: https://www.epfl.ch/schools/enac/about/open-science/open-research/trajnet/
- missing: `['terms_accepted_by_user', 'terms_acceptance_date', 'allowed_use', 'local_path', 'source_identity']`
- next user action: Confirm official terms and fill `trajnetplusplus_official` fields in source_terms_confirmation_template_stage42.json: terms_accepted_by_user, terms_acceptance_date, allowed_use, local_path, source_identity.
- next commands after confirmation: `['.venv-pytorch/bin/python run_stage42_source_terms_confirmation_validator.py', '.venv-pytorch/bin/python run_stage42_unified_guarded_conversion_queue.py', '.venv-pytorch/bin/python run_stage42_guarded_source_conversion_launcher.py']`
- claim guard: Do not count this as converted/evaluated until validator, guarded conversion, no-leakage, and source-CV all pass.

## 8. FW-TERMS-eth_biwi_original

- target: `eth_biwi_original`
- domain: `ETH_UCY`
- category: `legal_terms_and_local_path`
- priority: `59`
- status: `not_run_user_action_required`
- official_url: https://vision.ee.ethz.ch/datsets.html
- missing: `['terms_accepted_by_user', 'terms_acceptance_date', 'allowed_use', 'local_path', 'source_identity']`
- next user action: Confirm official terms and fill `eth_biwi_original` fields in source_terms_confirmation_template_stage42.json: terms_accepted_by_user, terms_acceptance_date, allowed_use, local_path, source_identity.
- next commands after confirmation: `['.venv-pytorch/bin/python run_stage42_source_terms_confirmation_validator.py', '.venv-pytorch/bin/python run_stage42_unified_guarded_conversion_queue.py', '.venv-pytorch/bin/python run_stage42_guarded_source_conversion_launcher.py']`
- claim guard: Do not count this as converted/evaluated until validator, guarded conversion, no-leakage, and source-CV all pass.
