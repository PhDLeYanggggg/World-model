# Numerical Fallback Sensitivity

Post-decision bounds only. No action is chosen from observed costs, and no original decision is changed.
For the affected query, sum absolute candidate-minus-CV error on eligible observed rows. Divide by the
site CV sum and by four sites and three seeds to bound any primary mean-gain change.
This deliberately overbounds an exact-count repair and retains all existing outcomes.

| View / predictor / policy | Eligible | Desired count | Max mean ADE gain shift pp |
|---|---:|---:|---:|
| deathCircle_seed29 / damped_velocity_005 / positive | 5 | None | 0.000171230 |
| hyang_seed43 / transformer / matched | 26 | 3 | 0.000946967 |
| hyang_seed43 / eqmotion / net | 29 | None | 0.000923990 |

The positive-policy failure also sets its matched-control reference count to zero at that query.
The matched Transformer arm is not globally exact-count matched because its own failed query falls to zero.
These limitations must accompany matched comparisons. Bounds do not certify risk or repair the solver.
