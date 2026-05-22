# Stage 16 Gates

Passed: 5 / 13

| gate | pass | evidence |
| --- | --- | --- |
| Data Gate | True | t50=433 |
| Oracle Label Gate | True | labels=595 |
| Failure Predictor Gate | False | AUROC=0.7346938775510204; ECE=0.13629626013901866 |
| Deterministic t+50 Gate | False | t50=0.009176227786756534 |
| Diagnostic t+100 Gate | False | t100_rows=81; imp=0.0114760269479769 |
| Hard/Failure Gate | False | hard=0.0114760269479769; failure=0.00789362952398513 |
| Easy Preservation Gate | True | easy_degradation=0.0 |
| Scene/Goal Gate | False | gain=0.0 |
| Interaction Gate | False | gain=0.0 |
| Physical Validity Gate | True | preserved_by_bounded_residual |
| Data Expansion Gate | True | user_action_required or data path found |
| Stage 5C Readiness Gate | False | Plan only; no latent execution in Stage16. |
| SMC Readiness Gate | False | Always false in Stage16. |

Do not enter Stage 5C. Oracle-distilled deterministic correction is not strong enough or t+100 remains diagnostic.

SMC remains disabled in Stage 16.
