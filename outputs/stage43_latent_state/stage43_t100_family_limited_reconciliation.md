# Stage43-BK T100 Family-Limited Reconciliation

- source: `fresh_stage43_bk_t100_family_limited_reconciliation`
- result_source: `fresh_reconciliation_from_stage43_p_t_u_bi_bj_verified_artifacts`
- verdict: `stage43_bk_t100_family_limited_reconciliation_pass`
- gate: `15 / 15`
- t100 family-limited ADE signal: `True`
- uniform t100 success: `False`
- t100 endpoint success: `False`

## Why This Reconciliation Exists

Stage43-BJ correctly kept the long objective active, but its `t100_raw_frame_diagnostic_not_solved` blocker was too coarse. Stage43-T/U show a small source-stable h100 full-waypoint ADE signal. The right boundary is not `t100 solved`; it is `family-limited t100 ADE diagnostic exists, while uniform t100 and endpoint-FDE success remain blocked`.

## Reference Candidate

- Stage43-P all ADE improvement: `50.25%`
- Stage43-P t50 ADE improvement: `51.23%`
- Stage43-P t100 raw-frame diagnostic: `0.00%`
- Stage43-P hard/failure: `47.88%`

## Family-Limited H100 Evidence

- Stage43-T source-stable h100 rows: `1440`
- Stage43-T full-waypoint ADE lift: `2.59%`
- Stage43-T full-waypoint ADE CI: `[2.06%, 3.14%]`
- Stage43-T endpoint FDE lift: `-0.55%`
- test sources: `['OpenTraj/datasets/TrajNet/Train/crowds/crowds_zara03.txt']`

## Integrated Policy Diagnostic

- integrated all ADE improvement: `50.28%`
- integrated t50 ADE improvement: `51.23%`
- integrated t100 raw-frame diagnostic: `0.18%`
- integrated t100 CI: `[0.14%, 0.22%]`
- integrated hard/failure: `47.91%`
- integrated easy degradation: `0.00%`
- t100 delta vs Stage43-P: `0.18%`

## H100 Slice Boundary

- h100 slice rows: `1440`
- h100 slice ADE lift: `2.59%`
- h100 slice endpoint FDE lift: `-0.55%`
- h100 slice hard/failure ADE lift: `2.59%`
- h100 slice easy degradation: `0.00%`

## Claim Update

- Allowed: report a family-limited raw-frame h100/t100 full-waypoint ADE diagnostic signal.
- Not allowed: report uniform t100 success.
- Not allowed: report h100/t100 endpoint-FDE success.
- Not allowed: metric, seconds-level, true-3D, foundation, Stage5C, or SMC claims.

## Gate

| gate | passed |
| --- | --- |
| `stage43_p_precondition_passed` | `True` |
| `stage43_t_precondition_passed` | `True` |
| `stage43_u_precondition_passed` | `True` |
| `stage43_bi_bj_preconditions_passed` | `True` |
| `integrated_policy_preserves_core_metrics` | `True` |
| `integrated_t100_improves_over_stage43_p` | `True` |
| `integrated_t100_ci_positive` | `True` |
| `h100_source_stable_ade_positive` | `True` |
| `h100_endpoint_blocker_explicit` | `True` |
| `uniform_t100_not_overclaimed` | `True` |
| `fresh_reconciliation_only` | `True` |
| `no_future_or_test_leakage` | `True` |
| `claim_boundary_not_overstated` | `True` |
| `stage5c_and_smc_false` | `True` |
| `long_objective_kept_active` | `True` |
