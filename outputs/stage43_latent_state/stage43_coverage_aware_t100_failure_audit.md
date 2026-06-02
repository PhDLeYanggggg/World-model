# Stage43-CH Coverage-Aware T100 Failure Audit

- source: `fresh_stage43_ch_coverage_aware_t100_failure_audit`
- result_source: `fresh_replay_audit_from_stage43_cg_medium_checkpoint`
- verdict: `stage43_ch_t100_failure_audit_pass_blocker_confirmed`
- gate: `11 / 11`
- replayed rows: `50000`

## Current Boundary

- This is a failure audit, not a new deployable t100 model.
- Dataset-local/raw-frame 2.5D only.
- No metric or seconds-level claim.
- Stage5C not executed; SMC not enabled.

## T100 Failure Confirmation

- t100 rows: `8443`
- t100 full-waypoint ADE improvement: `-5.51%`
- t100 endpoint FDE improvement: `-2.52%`
- t100 switch rate: `12.78%`
- t100 easy degradation: `7.00%`
- t100 bootstrap CI: `[-6.18%, -4.85%]`

## Horizon Slices

| horizon | rows | ADE improvement | endpoint improvement | switch | easy degradation |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10 | 15385 | `71.83%` | `74.96%` | `97.71%` | `0.00%` |
| 25 | 13913 | `60.96%` | `68.43%` | `87.06%` | `0.00%` |
| 50 | 12259 | `31.13%` | `41.49%` | `60.38%` | `0.00%` |
| 100 | 8443 | `-5.51%` | `-2.52%` | `12.78%` | `7.00%` |

## T100 Switch Attribution

| slice | rows | ADE improvement | switch | mean floor ADE | mean selected ADE | harm over floor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| t100_all | 8443 | `-5.51%` | `12.78%` | `0.125368` | `0.132279` | `0.006911` |
| t100_switched | 1079 | `-34.37%` | `100.00%` | `0.157332` | `0.211410` | `0.054077` |
| t100_fallback | 7364 | `0.00%` | `0.00%` | `0.120684` | `0.120684` | `0.000000` |
| t100_hard_failure | 8443 | `-5.51%` | `12.78%` | `0.125368` | `0.132279` | `0.006911` |
| t100_easy | 1903 | `-7.00%` | `6.15%` | `0.041592` | `0.044503` | `0.002911` |

## Domain Slices

| domain | rows | ADE improvement | t100? | switch | easy degradation |
| --- | ---: | ---: | --- | ---: | ---: |
| TrajNet | 28683 | `49.82%` | mixed | `73.60%` | `0.00%` |
| ETH_UCY | 15575 | `54.86%` | mixed | `66.25%` | `0.73%` |
| UCY | 5742 | `51.70%` | mixed | `73.09%` | `0.00%` |

## Interpretation

t100 raw-frame full-waypoint ADE improvement remains negative after CE medium replay; keep t100 floor and train a dedicated long-horizon repair before any t100 deployment claim.

Recommended next step: Train a t100-specific coverage-aware long-horizon head or per-horizon policy with validation-selected t100 safety constraints; do not change all/t50 deployment claims based on t100.

## Gate

| gate | passed |
| --- | --- |
| `cg_medium_precondition_present` | `True` |
| `t100_rows_present` | `True` |
| `t100_negative_confirmed` | `True` |
| `t100_ci_reported` | `True` |
| `domain_slice_reported` | `True` |
| `source_slice_reported` | `True` |
| `switched_vs_fallback_reported` | `True` |
| `prediction_diagnostics_reported` | `True` |
| `no_future_or_test_leakage` | `True` |
| `no_metric_seconds_stage5c_smc_claim` | `True` |
| `long_objective_kept_active` | `True` |
