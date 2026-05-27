# Stage42-FT Unified Guarded Conversion Queue

- source: `fresh_stage42_unified_guarded_conversion_queue`
- generated_at_utc: `2026-05-27T12:14:19.591887+00:00`
- gate: `12 / 12`
- verdict: `stage42_ft_unified_guarded_conversion_queue_pass`
- source_ready_targets: `0`
- h100_ready_candidates: `0`
- unified_queue_count: `0`
- blocked_action_count: `11`

## Queue

- Unified queue is empty because no global source target and no UCY H100 candidate is terms-ready.

## Blocked Actions

| scope | dataset | candidate | blockers | next action |
| --- | --- | --- | --- | --- |
| `global_source_manifest` | `ucy_crowd_original` | `ucy_crowd_original` | manual_terms_or_application_required, no_independent_t50_candidate, terms_not_accepted, terms_acceptance_date_missing, allowed_use_missing, local_path_confirmation_missing, source_identity_missing | fill explicit official terms/path/source-identity confirmation before conversion |
| `global_source_manifest` | `eth_biwi_original` | `eth_biwi_original` | manual_terms_or_application_required, no_independent_t50_candidate, terms_not_accepted, terms_acceptance_date_missing, allowed_use_missing, local_path_confirmation_missing, source_identity_missing | fill explicit official terms/path/source-identity confirmation before conversion |
| `global_source_manifest` | `trajnetplusplus_official` | `trajnetplusplus_official` | manual_terms_or_application_required, no_independent_t50_candidate, terms_not_accepted, terms_acceptance_date_missing, allowed_use_missing, local_path_confirmation_missing, source_identity_missing | fill explicit official terms/path/source-identity confirmation before conversion |
| `global_source_manifest` | `opentraj_toolkit` | `opentraj_toolkit` | no_independent_t50_candidate, terms_not_accepted, terms_acceptance_date_missing, allowed_use_missing, local_path_confirmation_missing, source_identity_missing | fill explicit official terms/path/source-identity confirmation before conversion |
| `global_source_manifest` | `aerialmpt_or_other_topdown` | `aerialmpt_or_other_topdown` | local_path_missing, schema_not_parseable, manual_terms_or_application_required, no_independent_t50_candidate, terms_not_accepted, terms_acceptance_date_missing, allowed_use_missing, local_path_confirmation_missing, source_identity_missing | fill explicit official terms/path/source-identity confirmation before conversion |
| `ucy_h100_candidate` | `ucy_crowd_original` | `UCY_zara02::obsmat` | terms_not_accepted, terms_acceptance_date_missing, allowed_use_missing, redistribution_policy_unknown, derived_data_policy_unknown, local_path_confirmation_missing, source_identity_missing, confirmed_by_user_missing | fill outputs/stage42_long_research/ucy_h100_candidate_terms_template_stage42.json after official UCY terms/path/source identity confirmation |
| `ucy_h100_candidate` | `ucy_crowd_original` | `UCY_zara01::obsmat` | terms_not_accepted, terms_acceptance_date_missing, allowed_use_missing, redistribution_policy_unknown, derived_data_policy_unknown, local_path_confirmation_missing, source_identity_missing, confirmed_by_user_missing | fill outputs/stage42_long_research/ucy_h100_candidate_terms_template_stage42.json after official UCY terms/path/source identity confirmation |
| `ucy_h100_candidate` | `ucy_crowd_original` | `UCY_students03::obsmat_px` | terms_not_accepted, terms_acceptance_date_missing, allowed_use_missing, redistribution_policy_unknown, derived_data_policy_unknown, local_path_confirmation_missing, source_identity_missing, confirmed_by_user_missing | fill outputs/stage42_long_research/ucy_h100_candidate_terms_template_stage42.json after official UCY terms/path/source identity confirmation |
| `ucy_h100_candidate` | `ucy_crowd_original` | `UCY_students03::obsmat` | terms_not_accepted, terms_acceptance_date_missing, allowed_use_missing, redistribution_policy_unknown, derived_data_policy_unknown, local_path_confirmation_missing, source_identity_missing, confirmed_by_user_missing | fill outputs/stage42_long_research/ucy_h100_candidate_terms_template_stage42.json after official UCY terms/path/source identity confirmation |
| `ucy_h100_candidate` | `ucy_crowd_original` | `UCY_students01::students001` | terms_not_accepted, terms_acceptance_date_missing, allowed_use_missing, redistribution_policy_unknown, derived_data_policy_unknown, local_path_confirmation_missing, source_identity_missing, confirmed_by_user_missing | fill outputs/stage42_long_research/ucy_h100_candidate_terms_template_stage42.json after official UCY terms/path/source identity confirmation |
| `ucy_h100_candidate` | `ucy_crowd_original` | `UCY_students03::students003` | terms_not_accepted, terms_acceptance_date_missing, allowed_use_missing, redistribution_policy_unknown, derived_data_policy_unknown, local_path_confirmation_missing, source_identity_missing, confirmed_by_user_missing | fill outputs/stage42_long_research/ucy_h100_candidate_terms_template_stage42.json after official UCY terms/path/source identity confirmation |

## Interpretation

- FT unifies the existing global source-conversion queue and the candidate-level UCY H100 queue.
- It is intentionally non-executing: empty queue means no conversion; non-empty queue still requires a later guarded parser/no-leakage/source-CV stage.
- No raw data, cache, converted dataset, metric/seconds claim, Stage5C, or SMC is produced.
