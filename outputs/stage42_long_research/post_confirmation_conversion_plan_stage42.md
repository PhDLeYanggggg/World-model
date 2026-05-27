# Stage42-GF Post-Confirmation Conversion Plan

- source: `fresh_stage42_gf_post_confirmation_conversion_plan`
- generated_at_utc: `2026-05-27T12:01:54.645899+00:00`
- git_commit: `dc50b07`
- input_hash: `3d7c29b4e6907719041b6ca0f9b3a7a25c3f3af7765e77065bd8730b5fe77153`
- gate: `16 / 16`
- verdict: `stage42_gf_post_confirmation_conversion_plan_pass`

## Role

- Turns GE `conversion_capability_prefill` into a ranked source-level post-confirmation execution plan.
- It is not legal permission, not a conversion queue, not converted data, and not evaluation.
- It tells the next guarded conversion stage which source rows become worth converting after user-confirmed terms/path/source identity.

## Summary

- planned_source_rows: `6`
- technical_ready_after_terms_sources: `5`
- source_cv_feasible_after_terms_datasets: `1`
- t50/t100 windows after terms: `10060` / `5696`
- source_ready_now: `0`
- conversion_ready_targets_in_manifest: `0`

## Dataset Summary

| dataset | domain | sources | tech-ready after terms | source-CV after terms | t50 | t100 | top source |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `ucy_crowd_original` | `UCY` | 4 | 3 | True | 9554 | 5605 | `UCY_students03` |
| `eth_biwi_original` | `ETH_UCY` | 2 | 2 | False | 506 | 91 | `ETH_seq_eth` |

## Top Source Plan Rows

| rank | dataset | domain | source | score | t50 | t100 | tech after terms | source ready now | missing user fields |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `ucy_crowd_original` | `UCY` | `UCY_students03` | 14442.0 | 6491 | 3413 | True | False | terms_accepted_by_user, terms_acceptance_date, allowed_use, local_path, source_identity, confirmed_by_user, redistribution_allowed, derived_data_allowed |
| 2 | `ucy_crowd_original` | `UCY` | `UCY_zara02` | 8138.0 | 2823 | 2095 | True | False | terms_accepted_by_user, terms_acceptance_date, allowed_use, local_path, source_identity, confirmed_by_user, redistribution_allowed, derived_data_allowed |
| 3 | `ucy_crowd_original` | `UCY` | `UCY_zara01` | 1559.0 | 240 | 97 | True | False | terms_accepted_by_user, terms_acceptance_date, allowed_use, local_path, source_identity, confirmed_by_user, redistribution_allowed, derived_data_allowed |
| 4 | `ucy_crowd_original` | `UCY` | `UCY_zara03` | 1025.0 | 0 | 0 | False | False | terms_accepted_by_user, terms_acceptance_date, allowed_use, local_path, source_identity, confirmed_by_user, redistribution_allowed, derived_data_allowed |
| 5 | `eth_biwi_original` | `ETH_UCY` | `ETH_seq_eth` | 598.0 | 291 | 91 | True | False | terms_accepted_by_user, terms_acceptance_date, allowed_use, local_path, source_identity, confirmed_by_user, redistribution_allowed, derived_data_allowed |
| 6 | `eth_biwi_original` | `ETH_UCY` | `ETH_seq_hotel` | 340.0 | 215 | 0 | True | False | terms_accepted_by_user, terms_acceptance_date, allowed_use, local_path, source_identity, confirmed_by_user, redistribution_allowed, derived_data_allowed |

## Claim Boundary

- This plan is intentionally non-executing.
- `source_ready_now` remains zero because user confirmation is still absent.
- It does not permit metric/seconds, true-3D, foundation, Stage5C, or SMC claims.
