# Stage43-AV Source-Horizon Safety Envelope

- source: `fresh_stage43_av_source_horizon_safety_envelope`
- result_source: `fresh_test_diagnostic_safety_envelope_over_stage43_au_trials`
- verdict: `stage43_av_source_horizon_safety_envelope_pass`
- gate: `12 / 12`
- trial count: `30`
- test diagnostic only: `True`
- deploy new policy: `False`

## Envelope Summary

- aggregate-safe trial count: `29`
- domain-easy-safe trial count: `19`
- deployable-like trial count: `6`
- best t50 diagnostic trial: `all_guard0.05_eth0.40_traj0.35`
- best t50 diagnostic metrics: all `27.64%`, t50 `15.06%`, hard `29.44%`, easy `2.13%`

Best deployable-like diagnostic trial: `t50_guard0.03_eth0.15_traj0.10`
- metrics: all `2.58%`, t50 `12.80%`, hard `2.87%`, easy `0.00%`

## Trial Diagnostic Table

| trial | focus | guard | ETH cap | Traj cap | test all | test t50 | test hard | test easy | domain easy safe | positive t50 domains | negative sources | deployable-like |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| `all_guard0.01_eth0.15_traj0.10` | `all` | `0.01` | `0.15` | `0.1` | `17.78%` | `8.85%` | `19.06%` | `0.00%` | `True` | `1` | `1` | `False` |
| `all_guard0.01_eth0.25_traj0.10` | `all` | `0.01` | `0.25` | `0.1` | `17.78%` | `8.85%` | `19.06%` | `0.00%` | `True` | `1` | `1` | `False` |
| `all_guard0.01_eth0.35_traj0.10` | `all` | `0.01` | `0.35` | `0.1` | `17.78%` | `8.85%` | `19.06%` | `0.00%` | `True` | `1` | `1` | `False` |
| `all_guard0.01_eth0.15_traj0.20` | `all` | `0.01` | `0.15` | `0.2` | `18.38%` | `8.85%` | `20.03%` | `0.47%` | `False` | `1` | `1` | `False` |
| `all_guard0.01_eth0.25_traj0.25` | `all` | `0.01` | `0.25` | `0.25` | `18.38%` | `8.85%` | `20.03%` | `0.47%` | `False` | `1` | `1` | `False` |
| `all_guard0.01_eth0.40_traj0.35` | `all` | `0.01` | `0.4` | `0.35` | `18.38%` | `8.85%` | `20.03%` | `0.47%` | `False` | `1` | `1` | `False` |
| `all_guard0.03_eth0.15_traj0.10` | `all` | `0.03` | `0.15` | `0.1` | `23.11%` | `11.36%` | `24.41%` | `0.00%` | `True` | `1` | `1` | `False` |
| `all_guard0.03_eth0.25_traj0.10` | `all` | `0.03` | `0.25` | `0.1` | `23.13%` | `12.05%` | `24.43%` | `0.00%` | `True` | `2` | `1` | `False` |
| `all_guard0.03_eth0.35_traj0.10` | `all` | `0.03` | `0.35` | `0.1` | `23.13%` | `12.05%` | `24.43%` | `0.00%` | `True` | `2` | `1` | `False` |
| `all_guard0.03_eth0.15_traj0.20` | `all` | `0.03` | `0.15` | `0.2` | `23.85%` | `11.36%` | `25.51%` | `0.00%` | `False` | `1` | `1` | `False` |
| `all_guard0.03_eth0.25_traj0.25` | `all` | `0.03` | `0.25` | `0.25` | `24.32%` | `12.05%` | `26.13%` | `0.08%` | `False` | `2` | `1` | `False` |
| `all_guard0.03_eth0.40_traj0.35` | `all` | `0.03` | `0.4` | `0.35` | `24.57%` | `12.80%` | `26.44%` | `0.50%` | `False` | `3` | `1` | `False` |
| `all_guard0.05_eth0.15_traj0.10` | `all` | `0.05` | `0.15` | `0.1` | `24.13%` | `12.35%` | `25.43%` | `0.00%` | `True` | `1` | `1` | `False` |
| `all_guard0.05_eth0.25_traj0.10` | `all` | `0.05` | `0.25` | `0.1` | `25.90%` | `13.79%` | `27.05%` | `0.00%` | `False` | `2` | `1` | `False` |
| `all_guard0.05_eth0.35_traj0.10` | `all` | `0.05` | `0.35` | `0.1` | `25.90%` | `13.79%` | `27.05%` | `0.00%` | `False` | `2` | `1` | `False` |
| `all_guard0.05_eth0.15_traj0.20` | `all` | `0.05` | `0.15` | `0.2` | `24.89%` | `12.35%` | `26.53%` | `0.00%` | `False` | `1` | `1` | `False` |
| `all_guard0.05_eth0.25_traj0.25` | `all` | `0.05` | `0.25` | `0.25` | `27.28%` | `13.79%` | `28.89%` | `0.30%` | `False` | `2` | `1` | `False` |
| `all_guard0.05_eth0.40_traj0.35` | `all` | `0.05` | `0.4` | `0.35` | `27.64%` | `15.06%` | `29.44%` | `2.13%` | `False` | `3` | `1` | `False` |
| `t50_guard0.01_eth0.15_traj0.10` | `t50` | `0.01` | `0.15` | `0.1` | `1.78%` | `8.85%` | `1.98%` | `0.00%` | `True` | `1` | `0` | `False` |
| `t50_guard0.01_eth0.25_traj0.10` | `t50` | `0.01` | `0.25` | `0.1` | `1.78%` | `8.85%` | `1.98%` | `0.00%` | `True` | `1` | `0` | `False` |
| `t50_guard0.01_eth0.35_traj0.10` | `t50` | `0.01` | `0.35` | `0.1` | `1.78%` | `8.85%` | `1.98%` | `0.00%` | `True` | `1` | `0` | `False` |
| `t50_guard0.01_eth0.15_traj0.20` | `t50` | `0.01` | `0.15` | `0.2` | `1.78%` | `8.85%` | `1.98%` | `0.00%` | `True` | `1` | `0` | `False` |
| `t50_guard0.01_eth0.25_traj0.25` | `t50` | `0.01` | `0.25` | `0.25` | `1.78%` | `8.85%` | `1.98%` | `0.00%` | `True` | `1` | `0` | `False` |
| `t50_guard0.01_eth0.40_traj0.35` | `t50` | `0.01` | `0.4` | `0.35` | `1.78%` | `8.85%` | `1.98%` | `0.00%` | `True` | `1` | `0` | `False` |
| `t50_guard0.03_eth0.15_traj0.10` | `t50` | `0.03` | `0.15` | `0.1` | `2.58%` | `12.80%` | `2.87%` | `0.00%` | `True` | `3` | `0` | `True` |
| `t50_guard0.03_eth0.25_traj0.10` | `t50` | `0.03` | `0.25` | `0.1` | `2.58%` | `12.80%` | `2.87%` | `0.00%` | `True` | `3` | `0` | `True` |
| `t50_guard0.03_eth0.35_traj0.10` | `t50` | `0.03` | `0.35` | `0.1` | `2.58%` | `12.80%` | `2.87%` | `0.00%` | `True` | `3` | `0` | `True` |
| `t50_guard0.03_eth0.15_traj0.20` | `t50` | `0.03` | `0.15` | `0.2` | `2.58%` | `12.80%` | `2.87%` | `0.00%` | `True` | `3` | `0` | `True` |
| `t50_guard0.03_eth0.25_traj0.25` | `t50` | `0.03` | `0.25` | `0.25` | `2.58%` | `12.80%` | `2.87%` | `0.00%` | `True` | `3` | `0` | `True` |
| `t50_guard0.03_eth0.40_traj0.35` | `t50` | `0.03` | `0.4` | `0.35` | `2.58%` | `12.80%` | `2.87%` | `0.00%` | `True` | `3` | `0` | `True` |

## Interpretation

train source/horizon-specific safety heads or source-family expert, not another global cap, because test-diagnostic envelope shows aggregate lift and per-domain/source safety are not aligned

This audit does not freeze or deploy a policy. It uses held-out test only to map the safety envelope after Stage43-AU exposed a validation/test safety mismatch.

## Claim Boundary

- Dataset-local/raw-frame 2.5D only.
- No metric/seconds, true 3D, foundation, Stage5C, or SMC claim.
- Test diagnostics are not deployment threshold selection.

## Gate

| gate | passed |
| --- | --- |
| `stage43_at_precondition_passed` | `True` |
| `stage43_au_precondition_passed` | `True` |
| `stage43_k_precondition_passed` | `True` |
| `all_trials_audited` | `True` |
| `test_diagnostic_not_deployment_selection` | `True` |
| `aggregate_safe_trials_exist` | `True` |
| `domain_easy_safety_checked` | `True` |
| `deployable_like_count_reported` | `True` |
| `best_t50_trial_reported_with_flags` | `True` |
| `next_required_action_recorded` | `True` |
| `no_metric_seconds_stage5c_smc_claim` | `True` |
| `long_objective_kept_active` | `True` |
