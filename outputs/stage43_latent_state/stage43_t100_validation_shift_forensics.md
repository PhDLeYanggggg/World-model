# Stage43-CN T100 Validation/Test Shift Forensics

- source: `fresh_stage43_cn_t100_validation_shift_forensics`
- result_source: `fresh_current_matrix_t100_validation_test_shift_forensics`
- gate: `11 / 11`
- verdict: `stage43_cn_t100_validation_shift_forensics_pass_ucy_shift_blocker`
- deploy t100: `False`

## Replayed Selected Model

- target: `residual`
- train filter: `t50t100`
- l2: `10000.0`
- train rows: `58845`
- model hash matches Stage43-CM: `True`

## Raw Validation-Allowed Test Metrics

- all ADE lift: `-0.59%`
- t100 raw-frame diagnostic: `-3.86%`
- hard/failure lift: `-0.71%`
- easy degradation: `2.26%`
- switch rate: `17.24%`

## Validation/Test Source Overlap

- validation h100 rows: `15296`
- test h100 rows: `18070`
- source-file jaccard: `0.0000`
- scene jaccard: `0.0000`

## Source-Family Shift

| family | val rows | test rows | val lift | test lift | lift drop | val easy harm | test easy harm | reason |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ETH_UCY | 2560 | 0 | -1905.05% | 0.00% | 1905.05% | 11027.25% | 0.00% | `not_validation_allowed` |
| TrajNet_biwi | 0 | 1160 | 0.00% | -497.30% | -497.30% | 0.00% | 3007.96% | `not_validation_allowed` |
| TrajNet_crowds | 5608 | 1440 | 2.40% | 0.80% | -1.60% | 3.33% | 18.93% | `not_validation_allowed` |
| UCY | 7128 | 15470 | 1.54% | -4.32% | -5.86% | 0.00% | 21.00% | `validation_allowed_but_test_negative_or_easy_harm` |

## Root Causes

- `UCY_test_easy_harm`
- `UCY_test_lift_nonpositive`
- `low_val_test_scene_overlap`
- `low_val_test_source_file_overlap`
- `validation_allowed_family_failed_current_test`

## Worst Test Source Files

| source_file | rows | t100 lift | easy harm | mean floor ADE |
| --- | ---: | ---: | ---: | ---: |
| `/Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/TrajNet/Train/biwi/biwi_hotel.txt` | 1160 | -497.30% | 3007.96% | 0.0825 |
| `/Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/UCY/students03/obsmat.txt` | 15470 | -4.32% | 21.00% | 0.1552 |
| `/Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/TrajNet/Train/crowds/crowds_zara03.txt` | 1440 | 0.80% | 18.93% | 0.1298 |

## Interpretation

Stage43-CM did not fail because the current matrix lacked t100 rows. It failed because a validation-positive UCY source-family rule did not generalize to current test: the same rule becomes negative on t100 and harms easy rows. This makes source-file/scene-level validation support a required next constraint before any t100 switch can be deployed.

Boundary unchanged: dataset-local/raw-frame 2.5D only; no metric or seconds-level claim; no Stage5C execution; no SMC.
