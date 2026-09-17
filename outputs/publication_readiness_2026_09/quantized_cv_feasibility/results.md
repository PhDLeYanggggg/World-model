# Conditional Hidden-CV Feasibility

Fresh fit-only linear programs; no predictor training or model selection.
Each recorded inferred integer pixel is allowed a closed +/-0.501 pixel box under supplied H.
The fitted diagnostic line has constant velocity in dataset-local coordinates, not image coordinates.
All eight past and twelve future labels constrain this diagnostic oracle; none enters model inputs.

| Source | Label-side subset | Rows | Feasible | Infeasible | Not run | Infeasible CV-error share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| eth_eth | all_stationary | 81 | 35 | 46 | 0 | 98.2968% |
| eth_eth | changed | 59 | 13 | 46 | 0 | 98.2968% |
| eth_eth | unchanged | 22 | 22 | 0 | 0 | null |
| eth_eth | changed_returned_origin | 0 | 0 | 0 | 0 | null |
| eth_hotel | all_stationary | 284 | 169 | 115 | 0 | 97.2134% |
| eth_hotel | changed | 129 | 14 | 115 | 0 | 97.2134% |
| eth_hotel | unchanged | 155 | 155 | 0 | 0 | null |
| eth_hotel | changed_returned_origin | 37 | 0 | 37 | 0 | 100.0000% |

Rows overlap and are not independent observations. Agent/run counts and solver statuses are in audit.json.
Infeasibility rules out only this hidden-CV plus rounding explanation. It does not prove annotation error,
human intent, true acceleration, or that a different model cannot learn a predictor.
Feasibility is also not proof of physical motion or an available inference-time velocity.
No exclusions, new target labels, threshold tuning, development/calibration/confirmation access,
metric/seconds claims, deployment, Stage5C or SMC. Original failed predictors remain failed.
