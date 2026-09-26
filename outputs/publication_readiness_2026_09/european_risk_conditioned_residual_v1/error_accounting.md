# Descriptive Error Accounting

For old error e and applied shift d, MSE change = 2 mean(e*d) + mean(d^2).
All 1,728 identities agree with direct arithmetic. No shrinkage or other parameter is fitted from these outcomes.
This explains the implemented error difference algebraically, not its causal origin.

| Inputs / bank / arm / control | Improved | Unchanged | Non-helpful aggregate direction | Helpful direction, excess size |
|---|---:|---:|---:|---:|
| full / oof / risk_only / original | 33 | 0 | 24 | 15 |
| full / oof / risk_only / common_context | 27 | 0 | 18 | 27 |
| full / oof / risk_context / original | 39 | 0 | 14 | 19 |
| full / oof / risk_context / common_context | 38 | 0 | 21 | 13 |
| full / in_sample_next / risk_only / original | 41 | 0 | 19 | 12 |
| full / in_sample_next / risk_only / common_context | 39 | 0 | 23 | 10 |
| full / in_sample_next / risk_context / original | 53 | 0 | 8 | 11 |
| full / in_sample_next / risk_context / common_context | 43 | 0 | 23 | 6 |
| full / in_sample_prev / risk_only / original | 46 | 0 | 18 | 8 |
| full / in_sample_prev / risk_only / common_context | 39 | 0 | 21 | 12 |
| full / in_sample_prev / risk_context / original | 52 | 0 | 8 | 12 |
| full / in_sample_prev / risk_context / common_context | 43 | 0 | 20 | 9 |
| motion_only / oof / risk_only / original | 50 | 0 | 11 | 11 |
| motion_only / oof / risk_only / common_context | 38 | 0 | 22 | 12 |
| motion_only / oof / risk_context / original | 41 | 0 | 16 | 15 |
| motion_only / oof / risk_context / common_context | 48 | 0 | 19 | 5 |
| motion_only / in_sample_next / risk_only / original | 47 | 0 | 24 | 1 |
| motion_only / in_sample_next / risk_only / common_context | 40 | 0 | 28 | 4 |
| motion_only / in_sample_next / risk_context / original | 40 | 0 | 19 | 13 |
| motion_only / in_sample_next / risk_context / common_context | 39 | 0 | 28 | 5 |
| motion_only / in_sample_prev / risk_only / original | 46 | 0 | 18 | 8 |
| motion_only / in_sample_prev / risk_only / common_context | 41 | 0 | 24 | 7 |
| motion_only / in_sample_prev / risk_context / original | 49 | 0 | 14 | 9 |
| motion_only / in_sample_prev / risk_context / common_context | 46 | 0 | 19 | 7 |

Each row contains 72 dependent seed/locality views; these counts are not independent replications.
Post-freeze source-development diagnosis only. No trajectory, deployment, metric/seconds or physical-safety claim.
