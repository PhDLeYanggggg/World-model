# Stage43-AU Domain Failure Repair Attempt

- source: `fresh_stage43_au_domain_failure_repair`
- result_source: `fresh_validation_selected_domain_horizon_repair_trials`
- verdict: `stage43_au_domain_failure_repair_attempt_pass`
- gate: `12 / 12`
- trial count: `30`
- deployment decision: `keep_stage43_k_or_stage43_ao_floor_protected_candidate`

## Why This Stage Exists

Stage43-AT showed strong protected aggregate evidence but weak external t50 slices: ETH_UCY t50 was barely positive and TrajNet t50 was fallback-like. Stage43-AU runs a bounded validation-only repair attempt over domain caps, easy guards, t50 focus, hard/failure focus, Stage35 gain score, and domain-expert caps. Test is evaluated once after validation selection.

## Selected Policy

- name: `all_guard0.05_eth0.25_traj0.25`
- focus: `all`
- easy guard: `0.05`
- domain caps: `{'ETH_UCY': 0.25, 'TrajNet': 0.25, 'UCY': 1.0}`

## Test Metrics

- all improvement: `27.28%`
- t50 improvement: `13.79%`
- t100 raw-frame diagnostic: `0.48%`
- hard/failure improvement: `28.89%`
- easy degradation: `0.30%`
- switch rate: `25.23%`

## Delta Vs Stage43-K Source-Safe Repair

- all delta: `4.17%`
- t50 delta: `2.42%`
- hard/failure delta: `4.49%`
- easy degradation delta: `0.30%`
- per-domain easy safe: `False`
- nonpositive source count: `1`

## Per-Domain Test Metrics

| domain | rows | all | t50 | t100 raw | hard/failure | easy degradation | switch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ETH_UCY` | 70585 | `22.39%` | `2.06%` | `-2.39%` | `23.38%` | `4.59%` | `21.00%` |
| `TrajNet` | 9611 | `19.75%` | `0.00%` | `0.00%` | `25.26%` | `14.04%` | `24.99%` |
| `UCY` | 9540 | `56.57%` | `52.18%` | `19.38%` | `57.65%` | `0.00%` | `56.82%` |

## Remaining Blocked Slices After Attempt

| slice | metric | reason |
| --- | ---: | --- |
| `domain:ETH_UCY:easy` | `4.59%` | per-domain easy degradation exceeds 2%; policy cannot be deployed even if aggregate easy is safe |
| `domain:ETH_UCY:t100_raw` | `-2.39%` | t100 raw-frame diagnostic remains negative |
| `domain:TrajNet:easy` | `14.04%` | per-domain easy degradation exceeds 2%; policy cannot be deployed even if aggregate easy is safe |
| `domain:TrajNet:t50` | `0.00%` | t50 transfer is <= 1% under selected safe policy |
| `source:83b0417df499ccae:all` | `-10.07%` | source is safely floored or non-positive; not positive transfer |

## Trial Summary

| trial | focus | guard | ETH cap | Traj cap | val all | val t50 | val easy | safe |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `all_guard0.01_eth0.15_traj0.10` | `all` | `0.01` | `0.15` | `0.1` | `17.88%` | `7.28%` | `0.00%` | `True` |
| `all_guard0.01_eth0.25_traj0.10` | `all` | `0.01` | `0.25` | `0.1` | `20.64%` | `7.54%` | `0.00%` | `True` |
| `all_guard0.01_eth0.35_traj0.10` | `all` | `0.01` | `0.35` | `0.1` | `20.74%` | `7.91%` | `0.00%` | `True` |
| `all_guard0.01_eth0.15_traj0.20` | `all` | `0.01` | `0.15` | `0.2` | `23.57%` | `7.28%` | `0.00%` | `True` |
| `all_guard0.01_eth0.25_traj0.25` | `all` | `0.01` | `0.25` | `0.25` | `28.02%` | `10.06%` | `0.00%` | `True` |
| `all_guard0.01_eth0.40_traj0.35` | `all` | `0.01` | `0.4` | `0.35` | `28.18%` | `10.70%` | `0.00%` | `True` |
| `all_guard0.03_eth0.15_traj0.10` | `all` | `0.03` | `0.15` | `0.1` | `21.08%` | `11.37%` | `0.00%` | `True` |
| `all_guard0.03_eth0.25_traj0.10` | `all` | `0.03` | `0.25` | `0.1` | `23.90%` | `11.37%` | `0.00%` | `True` |
| `all_guard0.03_eth0.35_traj0.10` | `all` | `0.03` | `0.35` | `0.1` | `25.41%` | `13.15%` | `0.00%` | `False` |
| `all_guard0.03_eth0.15_traj0.20` | `all` | `0.03` | `0.15` | `0.2` | `26.28%` | `11.37%` | `0.00%` | `True` |
| `all_guard0.03_eth0.25_traj0.25` | `all` | `0.03` | `0.25` | `0.25` | `31.64%` | `11.37%` | `0.00%` | `True` |
| `all_guard0.03_eth0.40_traj0.35` | `all` | `0.03` | `0.4` | `0.35` | `35.90%` | `18.23%` | `0.00%` | `False` |
| `all_guard0.05_eth0.15_traj0.10` | `all` | `0.05` | `0.15` | `0.1` | `22.66%` | `13.46%` | `0.00%` | `True` |
| `all_guard0.05_eth0.25_traj0.10` | `all` | `0.05` | `0.25` | `0.1` | `25.40%` | `13.46%` | `0.00%` | `True` |
| `all_guard0.05_eth0.35_traj0.10` | `all` | `0.05` | `0.35` | `0.1` | `27.20%` | `14.63%` | `0.00%` | `False` |
| `all_guard0.05_eth0.15_traj0.20` | `all` | `0.05` | `0.15` | `0.2` | `27.78%` | `13.46%` | `0.00%` | `True` |
| `all_guard0.05_eth0.25_traj0.25` | `all` | `0.05` | `0.25` | `0.25` | `32.87%` | `13.46%` | `0.00%` | `True` |
| `all_guard0.05_eth0.40_traj0.35` | `all` | `0.05` | `0.4` | `0.35` | `38.45%` | `16.75%` | `0.00%` | `False` |
| `t50_guard0.01_eth0.15_traj0.10` | `t50` | `0.01` | `0.15` | `0.1` | `2.25%` | `10.70%` | `0.00%` | `True` |
| `t50_guard0.01_eth0.25_traj0.10` | `t50` | `0.01` | `0.25` | `0.1` | `2.25%` | `10.70%` | `0.00%` | `True` |
| `t50_guard0.01_eth0.35_traj0.10` | `t50` | `0.01` | `0.35` | `0.1` | `2.25%` | `10.70%` | `0.00%` | `True` |
| `t50_guard0.01_eth0.15_traj0.20` | `t50` | `0.01` | `0.15` | `0.2` | `2.25%` | `10.70%` | `0.00%` | `True` |
| `t50_guard0.01_eth0.25_traj0.25` | `t50` | `0.01` | `0.25` | `0.25` | `2.25%` | `10.70%` | `0.00%` | `True` |
| `t50_guard0.01_eth0.40_traj0.35` | `t50` | `0.01` | `0.4` | `0.35` | `2.25%` | `10.70%` | `0.00%` | `True` |
| `t50_guard0.03_eth0.15_traj0.10` | `t50` | `0.03` | `0.15` | `0.1` | `3.83%` | `18.23%` | `0.00%` | `False` |
| `t50_guard0.03_eth0.25_traj0.10` | `t50` | `0.03` | `0.25` | `0.1` | `3.83%` | `18.23%` | `0.00%` | `False` |
| `t50_guard0.03_eth0.35_traj0.10` | `t50` | `0.03` | `0.35` | `0.1` | `3.83%` | `18.23%` | `0.00%` | `False` |
| `t50_guard0.03_eth0.15_traj0.20` | `t50` | `0.03` | `0.15` | `0.2` | `3.83%` | `18.23%` | `0.00%` | `False` |
| `t50_guard0.03_eth0.25_traj0.25` | `t50` | `0.03` | `0.25` | `0.25` | `3.83%` | `18.23%` | `0.00%` | `False` |
| `t50_guard0.03_eth0.40_traj0.35` | `t50` | `0.03` | `0.4` | `0.35` | `3.83%` | `18.23%` | `0.00%` | `False` |

## Claim Boundary

- This is a bounded repair attempt, not a new deployment freeze.
- The selected policy is validation-selected; test is not used to tune thresholds.
- Dataset-local/raw-frame 2.5D only.
- No metric/seconds, true 3D, foundation, Stage5C, or SMC claim.

## Gate

| gate | passed |
| --- | --- |
| `stage43_at_precondition_passed` | `True` |
| `stage43_k_precondition_passed` | `True` |
| `bounded_trials_at_most_30` | `True` |
| `multiple_repair_levers_attempted` | `True` |
| `validation_only_selection_recorded` | `True` |
| `test_eval_completed` | `True` |
| `easy_preserved` | `True` |
| `all_nonnegative` | `True` |
| `weak_slices_explicitly_reported` | `True` |
| `deployment_not_overclaimed` | `True` |
| `no_metric_seconds_stage5c_smc_claim` | `True` |
| `long_objective_kept_active` | `True` |
