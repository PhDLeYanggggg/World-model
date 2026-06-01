# Stage43-AM Bounded Residual Statistical Confirmation

- source: `fresh_stage43_am_bounded_residual_statistical_confirmation`
- result_source: `fresh_bootstrap_confirmation_over_frozen_stage43_al_candidate`
- gate: `12 / 12`
- verdict: `stage43_am_bounded_residual_statistically_confirmed`
- statistically confirmed: `True`
- bootstrap n: `2000`
- stored policy replay max abs diff: `0.00000000`

## Bootstrap Delta vs Stored Hard Switch

| metric | low | mean | high | rows |
| --- | ---: | ---: | ---: | ---: |
| all_delta_improvement | `7.78%` | `8.23%` | `8.68%` | `16000` |
| t50_delta_improvement | `9.86%` | `10.52%` | `11.21%` | `3856` |
| t100_delta_improvement | `16.07%` | `17.80%` | `19.58%` | `3180` |
| hard_failure_delta_improvement | `8.47%` | `8.97%` | `9.45%` | `12510` |
| easy_degradation_bounded | `0.00%` | `0.00%` | `0.00%` | `4771` |
| easy_degradation_delta | `0.00%` | `0.00%` | `0.00%` | `4771` |

## Slice Delta Summary

| slice | rows | stored | bounded | delta | switch |
| --- | ---: | ---: | ---: | ---: | ---: |
| domain:ETH_UCY | `12648` | `29.80%` | `38.01%` | `8.21%` | `60.71%` |
| domain:TrajNet | `1678` | `9.18%` | `24.19%` | `15.02%` | `65.55%` |
| domain:UCY | `1674` | `43.51%` | `47.29%` | `3.78%` | `78.55%` |
| horizon:10 | `4782` | `50.61%` | `53.36%` | `2.75%` | `86.16%` |
| horizon:25 | `4182` | `40.58%` | `48.39%` | `7.81%` | `83.64%` |
| horizon:50 | `3856` | `16.45%` | `26.96%` | `10.52%` | `64.19%` |
| horizon:100 | `3180` | `-17.79%` | `0.00%` | `17.79%` | `0.00%` |

## Interpretation

- This confirms or rejects the Stage43-AL bounded-residual candidate using fresh replay plus bootstrap deltas over the frozen test rows.
- The bounded residual is still floor-protected and h100-guarded; global floor removal remains unsupported.
- Dataset-local/raw-frame 2.5D only; no metric/seconds claim; no Stage5C; no SMC.

## Gate

| gate | passed |
| --- | --- |
| stage43_al_candidate_available | `True` |
| stage43_m_exact_replay | `True` |
| feature_schema_and_rows_match | `True` |
| bootstrap_n_at_least_2000 | `True` |
| all_delta_ci_positive | `True` |
| t50_delta_ci_positive | `True` |
| hard_failure_delta_ci_positive | `True` |
| t100_delta_ci_nonnegative | `True` |
| easy_degradation_ci_safe | `True` |
| per_domain_slices_reported | `True` |
| no_future_or_test_leakage | `True` |
| no_metric_seconds_stage5c_smc_claim | `True` |
