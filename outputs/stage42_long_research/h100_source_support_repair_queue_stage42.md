# Stage42-FQ H100 Source-Support Repair Queue

- source: `fresh_stage42_h100_source_support_repair_queue`
- generated_at_utc: `2026-05-27T09:36:49.592560+00:00`
- gate: `15 / 15`
- verdict: `stage42_fq_h100_source_support_repair_queue_pass`
- input FP verdict: `stage42_fp_h100_source_support_audit_pass`
- weak keys: `['TrajNet|100', 'UCY|100']`
- local files scanned: `101`
- uniform horizon claim allowed: `False`
- verification: `{'runner': '.venv-pytorch/bin/python run_stage42_h100_source_support_repair_queue.py -> 15/15', 'focused_pytest': '.venv-pytorch/bin/python -m pytest tests/test_stage42_h100_source_support_repair_queue.py -> 4 passed', 'full_pytest': '.venv-pytorch/bin/python -m pytest tests -> 836 passed'}`

## Local Gap Summary

| domain | files | t100 files | independent t100 groups | short/non-t100 files |
| --- | ---: | ---: | ---: | ---: |
| `ETH_UCY` | 18 | 7 | 6 | 11 |
| `TrajNet` | 59 | 0 | 0 | 59 |
| `UCY` | 24 | 6 | 4 | 18 |

## `TrajNet|100`

- target bucket: `students`
- FP blockers: `['long_horizon_h100_context_still_insufficient', 'low_material_headroom', 'oracle_low_margin_ambiguous', 'single_or_sparse_validation_source_support', 'source_specific_easy_safety_ci_failure', 'validation_to_test_source_family_shift']`
- candidate count: `0`
- repair status: `hard_blocker_no_local_trajnet_h100_long_source`
- reason: local TrajNet files are short snippets and cannot provide raw-frame h100 source support

| candidate | family | match | max track | est h100 windows | license | status |
| --- | --- | ---: | ---: | ---: | --- | --- |

## `UCY|100`

- target bucket: `zara`
- FP blockers: `['gain_harm_policy_abstained_due_to_validation_safety', 'long_horizon_h100_context_still_insufficient', 'low_material_headroom', 'oracle_low_margin_ambiguous', 'single_or_sparse_validation_source_support', 'source_specific_easy_safety_ci_failure', 'validation_to_test_source_family_shift']`
- candidate count: `6`
- repair status: `candidate_support_exists_terms_unverified`
- reason: local candidate support exists but terms/license and conversion/no-leakage/source-CV are not confirmed

| candidate | family | match | max track | est h100 windows | license | status |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `UCY/zara02/obsmat.txt` | `zara` | True | 583 | 2095 | `local_path_present_terms_unverified` | `not_converted_not_evaluated` |
| `UCY/zara01/obsmat.txt` | `zara` | True | 197 | 97 | `local_path_present_terms_unverified` | `not_converted_not_evaluated` |
| `UCY/students01/students001.txt` | `students` | False | 352 | 1949 | `local_path_present_terms_unverified` | `not_converted_not_evaluated` |
| `UCY/students03/obsmat.txt` | `students` | False | 539 | 3413 | `local_path_present_terms_unverified` | `not_converted_not_evaluated` |
| `UCY/students03/obsmat_px.txt` | `students` | False | 540 | 3415 | `local_path_present_terms_unverified` | `not_converted_not_evaluated` |
| `UCY/students03/students003.txt` | `students` | False | 289 | 879 | `local_path_present_terms_unverified` | `not_converted_not_evaluated` |

## Interpretation

- FQ is a repair queue, not a conversion or model-training step.
- Local UCY h100 candidates can only become support after terms confirmation, conversion, no-leakage, and train-only source-CV.
- Local TrajNet files do not currently provide long raw h100 support; TrajNet|100 needs official longer sources or the uniform-horizon claim must remain blocked.
- No raw data is committed, no auto-download is performed, and no metric/seconds/Stage5C/SMC claim is made.
