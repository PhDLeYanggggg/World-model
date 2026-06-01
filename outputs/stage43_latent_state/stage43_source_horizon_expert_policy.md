# Stage43-AW Source-Horizon Expert Policy

- source: `fresh_stage43_aw_source_horizon_expert_policy`
- result_source: `fresh_validation_selected_source_horizon_expert_policy`
- verdict: `stage43_aw_source_horizon_expert_policy_pass`
- gate: `12 / 12`
- candidate for reviewer replay: `True`
- deploy without replay: `False`
- deployment decision: `candidate_requires_reviewer_replay_before_deployment`

## Policy

- base: Stage43-K source-family guarded non-t50 base
- t50 expert: `t50_guard0.03_eth0.15_traj0.10`
- selection rule: validation aggregate/domain/source safe, >=2 positive t50 validation domains, maximize 3*t50 + all + 0.5*hard + domain-count bonus
- eligible validation candidates: `12`

## Test Metrics

- all improvement: `23.40%`
- t50 improvement: `12.80%`
- t100 raw-frame diagnostic: `1.35%`
- hard/failure improvement: `24.73%`
- easy degradation: `0.00%`
- switch rate: `19.37%`

## Bootstrap CI

- all CI: `[23.04%, 23.71%]`
- t50 CI: `[12.10%, 13.48%]`
- hard/failure CI: `[24.35%, 25.12%]`
- easy degradation CI: `[0.00%, 0.00%]`

## Delta Vs Stage43-K

- all delta: `0.29%`
- t50 delta: `1.44%`
- hard/failure delta: `0.32%`
- easy degradation delta: `0.00%`

## Per-Domain Test Metrics

| domain | rows | all | t50 | t100 raw | hard/failure | easy degradation | switch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ETH_UCY` | 70585 | `19.43%` | `1.15%` | `-0.72%` | `20.61%` | `1.21%` | `15.77%` |
| `TrajNet` | 9611 | `8.09%` | `11.24%` | `0.00%` | `9.10%` | `0.00%` | `12.22%` |
| `UCY` | 9540 | `53.57%` | `47.53%` | `15.70%` | `54.79%` | `0.00%` | `53.21%` |

## Safety Flags

- domain easy safe: `True`
- negative source count: `0`
- positive t50 domain count: `3`
- deployable-like under test diagnostic: `True`

## Claim Boundary

- This is validation-selected and test-evaluated once, but still requires reviewer replay before any deployment update.
- Dataset-local/raw-frame 2.5D only.
- No metric/seconds, true 3D, foundation, Stage5C, or SMC claim.

## Gate

| gate | passed |
| --- | --- |
| `stage43_k_precondition_passed` | `True` |
| `stage43_av_precondition_passed` | `True` |
| `validation_candidates_evaluated` | `True` |
| `validation_only_selection` | `True` |
| `validation_eligible_candidate_exists` | `True` |
| `test_eval_completed` | `True` |
| `test_aggregate_safe` | `True` |
| `test_t50_positive` | `True` |
| `domain_easy_safe` | `True` |
| `source_negative_free` | `True` |
| `no_metric_seconds_stage5c_smc_claim` | `True` |
| `long_objective_kept_active` | `True` |
