# Risk and Fallback Diagnosis

Post-freeze descriptive diagnostics, not a threshold search. Counts are dependent role/seed views.
A predicted2% ratio is not a realized2% guarantee. Undefined zero-denominator ratios remain null.

| Candidate | Known row-views | Selected | Harmful selected | Missed beneficial | Selected zero-reference |
|---|---:|---:|---:|---:|---:|
| dimensionless | 1871532 | 271579 | 51955 | 1070475 | 0 |
| damped | 1871532 | 587516 | 64642 | 892431 | 0 |

| Candidate | Reason | Row-views |
|---|---|---:|
| dimensionless | stationary_last_step | 888 |
| dimensionless | nonpositive_predicted_gain | 371454 |
| dimensionless | all_risk_veto | 1163354 |
| dimensionless | easy_risk_veto | 103812 |
| dimensionless | selected | 274306 |
| damped | stationary_last_step | 888 |
| damped | nonpositive_predicted_gain | 389660 |
| damped | all_risk_veto | 679109 |
| damped | easy_risk_veto | 250504 |
| damped | selected | 593653 |

## Locality Risk Extremes

| Candidate | View/site | Predicted selected harm ratio | Realized selected harm ratio |
|---|---|---:|---:|
| dimensionless | single2_seed43_controller0_dimensionless/eu-locality-007 | 0.006786425597965717 | 0.12909725421244997 |
| dimensionless | single2_seed43_controller0_dimensionless/eu-locality-124 | 0.008547508157789707 | 0.11755514735694354 |
| dimensionless | single2_seed17_controller0_dimensionless/eu-locality-007 | 0.005507327616214752 | 0.09261006051808784 |
| dimensionless | single2_seed17_controller1_dimensionless/eu-locality-126 | 0.00828393641859293 | 0.08018251836365935 |
| dimensionless | single2_seed29_controller0_dimensionless/eu-locality-007 | 0.005047245882451534 | 0.07857211577121914 |
| dimensionless | single2_seed29_controller1_dimensionless/eu-locality-126 | 0.008857570588588715 | 0.0738746661284192 |
| dimensionless | single2_seed17_controller1_dimensionless/eu-locality-082 | 0.008218660950660706 | 0.07275832049743124 |
| dimensionless | single2_seed43_controller1_dimensionless/eu-locality-126 | 0.008291381411254406 | 0.0724426305977569 |
| damped | single0_seed17_controller1_damped/eu-locality-020 | 0.0029264851473271847 | 0.009750875422409499 |
| damped | single1_seed29_controller2_damped/eu-locality-074 | 0.006578222848474979 | 0.008940460095677213 |
| damped | single1_seed43_controller2_damped/eu-locality-074 | 0.006921178661286831 | 0.008717197578795135 |
| damped | single0_seed29_controller1_damped/eu-locality-020 | 0.0033129972871392965 | 0.00840021668038671 |
| damped | single1_seed17_controller2_damped/eu-locality-074 | 0.006490309257060289 | 0.008386782446820056 |
| damped | single2_seed43_controller1_damped/eu-locality-074 | 0.0055474163964390755 | 0.008352367669139622 |
| damped | single0_seed17_controller2_damped/eu-locality-119 | 0.007736661471426487 | 0.008037053898465199 |
| damped | single0_seed29_controller2_damped/eu-locality-007 | 0.005290763918310404 | 0.00759054478605161 |

These ratios diagnose selected-set magnitude mismatch, not independent calibration.
Do not rescue the model by selecting the best displayed seed/site or refitting thresholds on these outcomes.
