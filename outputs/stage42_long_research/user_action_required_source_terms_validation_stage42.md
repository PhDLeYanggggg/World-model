# User Action Required: Stage42-CG Terms Validation

No target is conversion-ready. Required actions:

## ucy_crowd_original

- official_url: https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data
- CF blockers: manual_terms_or_application_required, no_independent_t50_candidate
- confirmation blockers: terms_not_accepted, terms_acceptance_date_missing, allowed_use_missing, local_path_confirmation_missing, source_identity_missing
- action: fill explicit official terms/path/source-identity confirmation before conversion

## eth_biwi_original

- official_url: https://vision.ee.ethz.ch/datsets.html
- CF blockers: manual_terms_or_application_required, no_independent_t50_candidate
- confirmation blockers: terms_not_accepted, terms_acceptance_date_missing, allowed_use_missing, local_path_confirmation_missing, source_identity_missing
- action: fill explicit official terms/path/source-identity confirmation before conversion

## trajnetplusplus_official

- official_url: https://www.epfl.ch/schools/enac/about/open-science/open-research/trajnet/
- CF blockers: manual_terms_or_application_required, no_independent_t50_candidate
- confirmation blockers: terms_not_accepted, terms_acceptance_date_missing, allowed_use_missing, local_path_confirmation_missing, source_identity_missing
- action: fill explicit official terms/path/source-identity confirmation before conversion

## opentraj_toolkit

- official_url: https://github.com/crowdbotp/OpenTraj
- CF blockers: no_independent_t50_candidate
- confirmation blockers: terms_not_accepted, terms_acceptance_date_missing, allowed_use_missing, local_path_confirmation_missing, source_identity_missing
- action: fill explicit official terms/path/source-identity confirmation before conversion

## aerialmpt_or_other_topdown

- official_url: user_or_web_verified_official_url_required
- CF blockers: local_path_missing, schema_not_parseable, manual_terms_or_application_required, no_independent_t50_candidate
- confirmation blockers: terms_not_accepted, terms_acceptance_date_missing, allowed_use_missing, local_path_confirmation_missing, source_identity_missing
- action: fill explicit official terms/path/source-identity confirmation before conversion

Do not convert or evaluate any source until this validator reports conversion_ready targets and a later no-leakage conversion gate passes.
