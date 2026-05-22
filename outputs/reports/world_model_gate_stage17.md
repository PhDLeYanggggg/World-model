# Stage 17 Gates

Passed: 7 / 12

| gate | pass | evidence |
| --- | --- | --- |
| Oracle Selector Headroom Gate | True | oracle_t50=0.271291 |
| Selector Training Gate | True | selector_t50=0.081954; hard=0.040700 |
| Selector Regret Gate | True | regret=0.702371 |
| Correction Specialist Gate | False | incremental=0.000000 |
| Easy Preservation Gate | True | easy_degradation=0.000000 |
| Hard/Failure Gate | False | hard=0.040700; failure=0.084696 |
| Scene/Goal Contribution Gate | False | gain=0.000000 |
| Interaction Contribution Gate | True | gain=0.081954 |
| Physical Validity Gate | True | preserved |
| Data Expansion Gate | True | stage17_user_action_required generated if model gates fail |
| Stage 5C Readiness Gate | False | Stage17 selector/correction gates did not pass; plan only, no execution |
| SMC Readiness Gate | False | SMC remains disabled |

Do not enter Stage 5C. Baseline selector/correction specialist is not strong enough.

SMC remains disabled.
