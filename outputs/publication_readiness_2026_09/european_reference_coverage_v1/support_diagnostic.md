# Fixed-Floor Reference-Coverage Support

TRAIN-only label-aware oracle, not deployment or causal learnability evidence.
Exact original floor, eligibility and all148record/context views retained.
Feasible counts: {'original_ADE': {'budget_feasible': 146, 'positive_feasible': 146}, 'harm_control': {'budget_feasible': 147, 'positive_feasible': 147}, 'new': {'budget_feasible': 147, 'positive_feasible': 147}}.
Transitions versus controls: {'original_ADE': {'gained': 1, 'lost': 0}, 'harm_control': {'gained': 0, 'lost': 0}}.

| Context / recording | Original | Harm-aware | Reference coverage | New minimum easy H/R |
|---|---|---|---|---:|
| single1_seed43_controller2_dimensionless / eu-locality-082/51 | False | False | False | 1.9310455230250796 |
| single2_seed43_controller1_dimensionless / eu-locality-082/51 | False | True | True | 0.0 |

The risk ratio threshold remains0.02, not2. Empty risk is not_estimable.
Six unchanged models cannot repair their preceding failures. New forecasts need new causal costs
and independent calibration before deployment. No Stage5C/SMC, metric or seconds claim.
