# Stage42-FP H100 Weak-Horizon Source / Support Audit

- source: `fresh_stage42_h100_weak_horizon_source_support_audit`
- generated_at_utc: `2026-05-27T09:15:56.156994+00:00`
- gate: `15 / 15`
- verdict: `stage42_fp_h100_source_support_audit_pass`
- input FO verdict: `stage42_fo_gain_harm_specialist_pass_with_horizon_limit`
- h100 weak horizons: `['TrajNet|100', 'UCY|100']`
- blocker counts: `{'long_horizon_h100_context_still_insufficient': 2, 'low_material_headroom': 2, 'oracle_low_margin_ambiguous': 2, 'single_or_sparse_validation_source_support': 2, 'source_specific_easy_safety_ci_failure': 2, 'validation_to_test_source_family_shift': 2, 'gain_harm_policy_abstained_due_to_validation_safety': 1}`
- decision: `diagnostic_only_keep_stage42_fh_fi_with_horizon_limit`

## Reconstructed FO Global Metric vs Floor

- all improvement: `35.20%`
- t50 improvement: `29.03%`
- t100 raw-frame diagnostic improvement: `21.14%`
- hard/failure improvement: `33.35%`
- easy degradation: `-37.10%`

## `TrajNet|100`

- test rows: `5608`; val rows: `1160`
- test sources/scenes: `1` / `1`
- val sources/scenes: `1` / `1`
- shared sources/families: `0` / `0`
- FO applied policy: `{'key': 'TrajNet|100', 'mode': 'gain_harm_model', 'gain_min': 0.0, 'harm_max': 0.35, 'max_switch': 0.35, 'rows': 5608, 'switch_rows': 1962}`
- oracle improvement vs FH: `1.06%`
- low-margin share: `{'0.01': 0.9880527817403709, '0.025': 0.9909058487874465, '0.05': 0.9917974322396577}`
- candidate delta vs FH: `{'fh': 0.0, 'fc': 0.00016787715689681182, 'di': 0.006568494026876315, 'fa': 0.0068507424394096406, 'fb': 0.006619585410310824, 'floor': -0.22659707897842285}`
- blockers: `['long_horizon_h100_context_still_insufficient', 'low_material_headroom', 'oracle_low_margin_ambiguous', 'single_or_sparse_validation_source_support', 'source_specific_easy_safety_ci_failure', 'validation_to_test_source_family_shift']`
- next action: `add_train_only_h100_source_support_or_build_source_family_specific_validation_before_more_modeling`

### Test source rows

| source/scene | rows | robust | all | t50 | t100raw | hard | easy | weak reasons |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `/Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/TrajNet/Train/crowds/students003.txt` | 5608 | False | `18.98%` | `0.00%` | `18.98%` | `18.98%` | `3.31%` | `['easy_ci_exceeds_2pct']` |

### Test scene rows

| source/scene | rows | robust | all | t50 | t100raw | hard | easy | weak reasons |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `TrajNet_crowds` | 5608 | False | `18.98%` | `0.00%` | `18.98%` | `18.98%` | `3.31%` | `['easy_ci_exceeds_2pct']` |

## `UCY|100`

- test rows: `1440`; val rows: `1440`
- test sources/scenes: `1` / `1`
- val sources/scenes: `1` / `1`
- shared sources/families: `0` / `0`
- FO applied policy: `{'key': 'UCY|100', 'mode': 'keep_fm', 'rows': 1440, 'switch_rows': 0}`
- oracle improvement vs FH: `2.74%`
- low-margin share: `{'0.01': 0.8631944444444445, '0.025': 0.8916666666666667, '0.05': 0.9027777777777778}`
- candidate delta vs FH: `{'fh': 0.0, 'fc': 8.175780679475775e-05, 'di': 0.007844899993292187, 'fa': 0.008678564560764435, 'fb': 0.007877262903924653, 'floor': -0.36970871528114957}`
- blockers: `['gain_harm_policy_abstained_due_to_validation_safety', 'long_horizon_h100_context_still_insufficient', 'low_material_headroom', 'oracle_low_margin_ambiguous', 'single_or_sparse_validation_source_support', 'source_specific_easy_safety_ci_failure', 'validation_to_test_source_family_shift']`
- next action: `add_train_only_h100_source_support_or_build_source_family_specific_validation_before_more_modeling`

### Test source rows

| source/scene | rows | robust | all | t50 | t100raw | hard | easy | weak reasons |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `/Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/TrajNet/Train/crowds/crowds_zara03.txt` | 1440 | False | `27.76%` | `0.00%` | `27.76%` | `27.76%` | `-1.91%` | `['easy_ci_exceeds_2pct']` |

### Test scene rows

| source/scene | rows | robust | all | t50 | t100raw | hard | easy | weak reasons |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `UCY_crowds` | 1440 | False | `27.76%` | `0.00%` | `27.76%` | `27.76%` | `-1.91%` | `['easy_ci_exceeds_2pct']` |

## Interpretation

- Stage42-FP is diagnostic only. It does not train a new policy and does not promote uniform horizon robustness.
- The remaining h100 weak horizons should be treated as source/support/context blockers until a future validation-selected repair passes them without easy or proximity regression.
- No metric/seconds-level, true-3D, Stage5C, or SMC claim is made.
