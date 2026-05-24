# Stage38 External Generalization Eval

- source: `fresh_run`

| domain | status | all | t50 | hard | easy | note |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| UCY | heldout_test | 0.134825 | 0.084573 | 0.155434 | 0.000411 |  |
| ETH_UCY | blocker | 0.000000 | 0.000000 | 0.000000 | 0.000000 | no held-out test split for frozen Stage37 evaluation; cannot claim deployable generalization |
| TrajNet | blocker | 0.000000 | 0.000000 | 0.000000 | 0.000000 | no held-out test split for frozen Stage37 evaluation; cannot claim deployable generalization |
| OpenTraj_mixed | blocker | 0.000000 | 0.000000 | 0.000000 | 0.000000 | mixed-domain held-out test beyond UCY is not available without redefining frozen Stage37 split |

- positive domains: `['UCY']`
- If only UCY is positive, current model remains UCY-biased until ETH/TrajNet held-out tests are built.
