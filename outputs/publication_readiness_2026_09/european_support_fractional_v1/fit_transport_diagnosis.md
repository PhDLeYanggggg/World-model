# Fitting Versus Held-Locality Transport

Post-freeze descriptive diagnosis; not a new endpoint or selection rule.
Each row is a dependent seed/source/fold view with its own training-only easy cut.
Training uses three equally weighted localities. Held evaluation uses the fourth.
Held MSE is conditional on positive causal forecast disagreement. Structural-zero
rows contribute zero harm and zero harm prediction, so including them changes
absolute MSE but not its within-view relative gain. No independent replication claim.

| Pair | Views | Training improved | Held improved | Train improves, held does not | Neither improves | Both improve |
|---|---:|---:|---:|---:|---:|---:|
| full | 72 | 56 | 33 | 29 | 10 | 27 |
| motion_only | 72 | 40 | 34 | 21 | 17 | 19 |

Counts diagnose this fixed objective, not causal proof of a unique failure mechanism.
No future-input change, forecast fitting, C policy, deployment, Stage5C or SMC.
