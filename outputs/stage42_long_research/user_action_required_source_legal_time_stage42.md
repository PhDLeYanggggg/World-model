# User Action Required: Stage42-DO Source Legal/Time Closure

No external source is conversion-ready yet. To unlock restricted source-specific metric/time or t+100 claims, fill the confirmation fields below and rerun the guarded conversion/no-leakage/source-CV path.

## ucy_crowd_original

- official_url: https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data
- domain: `UCY`
- conversion_ready_now: `False`
- source_specific_metric_time_candidates: `['UCY_zara01', 'UCY_zara02', 'UCY_zara03', 'UCY_students03']`
- terms_blockers: `['terms_not_accepted', 'terms_acceptance_date_missing', 'allowed_use_missing', 'local_path_confirmation_missing', 'source_identity_missing']`
- cf_blockers: `['manual_terms_or_application_required', 'no_independent_t50_candidate']`
- domain_blockers: `['source_terms_confirmation_or_conversion_readiness_missing', 'train_only_t100_source_cv_support_missing', 'additional_t100_sources_needed=1', 'legal_terms_blocked_targets=ucy_crowd_original']`
- required_action: accept/confirm official terms, allowed use, acceptance date, local path, and source identity

## eth_biwi_original

- official_url: https://vision.ee.ethz.ch/datsets.html
- domain: `ETH_UCY`
- conversion_ready_now: `False`
- source_specific_metric_time_candidates: `['ETH_seq_eth', 'ETH_seq_hotel']`
- terms_blockers: `['terms_not_accepted', 'terms_acceptance_date_missing', 'allowed_use_missing', 'local_path_confirmation_missing', 'source_identity_missing']`
- cf_blockers: `['manual_terms_or_application_required', 'no_independent_t50_candidate']`
- domain_blockers: `['source_terms_confirmation_or_conversion_readiness_missing', 'train_only_t100_source_cv_support_missing', 'additional_t100_sources_needed=2', 'legal_terms_blocked_targets=eth_biwi_original']`
- required_action: accept/confirm official terms, allowed use, acceptance date, local path, and source identity

## trajnetplusplus_official

- official_url: https://www.epfl.ch/schools/enac/about/open-science/open-research/trajnet/
- domain: `TrajNet`
- conversion_ready_now: `False`
- source_specific_metric_time_candidates: `[]`
- terms_blockers: `['terms_not_accepted', 'terms_acceptance_date_missing', 'allowed_use_missing', 'local_path_confirmation_missing', 'source_identity_missing']`
- cf_blockers: `['manual_terms_or_application_required', 'no_independent_t50_candidate']`
- domain_blockers: `['source_terms_confirmation_or_conversion_readiness_missing', 'train_only_t100_source_cv_support_missing', 'additional_t100_sources_needed=1', 'source_specific_metric_time_calibration_missing', 'legal_terms_blocked_targets=trajnetplusplus_official']`
- required_action: accept/confirm official terms, allowed use, acceptance date, local path, and source identity

## opentraj_toolkit

- official_url: https://github.com/crowdbotp/OpenTraj
- domain: `OpenTraj`
- conversion_ready_now: `False`
- source_specific_metric_time_candidates: `[]`
- terms_blockers: `['terms_not_accepted', 'terms_acceptance_date_missing', 'allowed_use_missing', 'local_path_confirmation_missing', 'source_identity_missing']`
- cf_blockers: `['no_independent_t50_candidate']`
- domain_blockers: `[]`
- required_action: accept/confirm official terms, allowed use, acceptance date, local path, and source identity

## aerialmpt_or_other_topdown

- official_url: user_or_web_verified_official_url_required
- domain: `other_topdown`
- conversion_ready_now: `False`
- source_specific_metric_time_candidates: `[]`
- terms_blockers: `['terms_not_accepted', 'terms_acceptance_date_missing', 'allowed_use_missing', 'local_path_confirmation_missing', 'source_identity_missing']`
- cf_blockers: `['local_path_missing', 'schema_not_parseable', 'manual_terms_or_application_required', 'no_independent_t50_candidate']`
- domain_blockers: `[]`
- required_action: accept/confirm official terms, allowed use, acceptance date, local path, and source identity

Do not convert, evaluate, or make metric/seconds/t100 deployment claims until the validator reports conversion-ready sources and a later no-leakage/source-CV gate passes.
