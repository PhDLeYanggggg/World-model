# Stage42-GY UCY H100 Terms Prefill From Integrity Manifest

- source: `fresh_stage42_gy_ucy_h100_terms_prefill_from_integrity`
- generated_at_utc: `2026-05-27T15:15:29.845552+00:00`
- git_commit: `079c084`
- gate: `14 / 14`
- verdict: `stage42_gy_ucy_h100_terms_prefill_pass`

## Summary

- prefill_rows: `6`
- rows_with_hash: `6`
- rows_with_source_identity_suggestion: `6`
- target_family_rows: `2`
- t100_capable_rows: `6`
- legal_acceptance_fields_blank: `True`
- conversion_ready_now_count: `0`

## Prefill Rows

| path | sha256 | source identity suggestion | t100 windows | target | user must fill |
| --- | --- | --- | ---: | ---: | --- |
| `UCY/zara02/obsmat.txt` | `0b29bdacd54b9712...` | `UCY::zara02::obsmat` | 2095 | True | terms/path/source_identity/confirmed_by_user |
| `UCY/zara01/obsmat.txt` | `bd999ebcfbc50682...` | `UCY::zara01::obsmat` | 97 | True | terms/path/source_identity/confirmed_by_user |
| `UCY/students03/obsmat_px.txt` | `8286c024198b7890...` | `UCY::students03::obsmat_px` | 3415 | False | terms/path/source_identity/confirmed_by_user |
| `UCY/students03/obsmat.txt` | `d9fd7048eab7d3d9...` | `UCY::students03::obsmat` | 3413 | False | terms/path/source_identity/confirmed_by_user |
| `UCY/students01/students001.txt` | `a6d87f278d94136f...` | `UCY::students01::students001` | 1949 | False | terms/path/source_identity/confirmed_by_user |
| `UCY/students03/students003.txt` | `e25798b660634330...` | `UCY::students03::students003` | 879 | False | terms/path/source_identity/confirmed_by_user |

## Interpretation

- GY reduces ambiguity for manual UCY terms confirmation by carrying forward hash and source-identity suggestions.
- GY intentionally leaves legal acceptance, allowed use, local path, and user confirmation blank.
- This is still not conversion, not evaluation, not h100 repair, not metric evidence, and not seconds-level evidence.

## Gate

| gate | pass |
| --- | ---: |
| `gx_input_verified` | True |
| `prefill_rows_written` | True |
| `hashes_included` | True |
| `source_identity_suggestions_included` | True |
| `target_family_preserved` | True |
| `legal_acceptance_not_autofilled` | True |
| `agent_may_not_fill_legal_acceptance` | True |
| `no_conversion_ready_claim` | True |
| `no_download_conversion_eval` | True |
| `user_action_written` | True |
| `no_future_test_or_central_velocity_leakage` | True |
| `no_metric_seconds_overclaim` | True |
| `stage5c_false` | True |
| `smc_false` | True |
