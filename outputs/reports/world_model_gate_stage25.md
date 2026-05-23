# Stage 25 Gates

- gates passed: `4 / 12`
- Stage5C readiness: `False`
- SMC readiness: `False`

| gate | pass | evidence |
| --- | --- | --- |
| Gate 1: Forensics Gate | True | Selector failure analysis completed and root causes recorded. |
| Gate 2: Regret Selector Gate | False | Regret selector >=5% t50 or >=10% hard/failure. |
| Gate 3: Easy Preservation Gate | True | Selected fallback policy keeps easy degradation <=2%. |
| Gate 4: Harm Reduction Gate | True | Harm over fallback reduced compared with Stage24 hard selector. |
| Gate 5: Failure-Assisted Gate | False | Failure predictor improves selector over no-failure variant. |
| Gate 6: Hierarchical Selector Gate | False | Hierarchical split/horizon/agent policy improves over global regret selector. |
| Gate 7: Fallback Policy Gate | True | Selected policy preserves easy cases while minimizing harm. |
| Gate 8: Hard/Failure Gate | False | Hard or BaselineFailureBench improves >=10%. |
| Gate 9: Scene/Goal Gate | False | Scene/goal measurable selector gain. |
| Gate 10: Interaction Gate | False | Interaction measurable selector gain. |
| Gate 11: Stage 5C Readiness Gate | False | Keep false unless selector + hard/failure + correction pass later; do not execute Stage5C. |
| Gate 12: SMC Readiness Gate | False | Keep false. |
