# Multi-Action Cost and Decision Evidence

Derived from four completed and exactly replayed SOURCE-development studies.
This exporter verifies frozen aggregates; it does not train, infer trajectories,
re-bootstrap, select a model, or open independent evaluation roles.

## Fixed Inner-Held Readout

Gain is equal-locality known-label ADE improvement over the frozen floor.
Intervals are conditional3,000-locality bootstrap intervals, not independent
confirmation. Repeated record/head views are not independent recordings.

| Study | Policy | Gain % [95% CI] | Supported record/head views | Undefined easy views | Interventions (unknown) |
|---|---|---:|---:|---:|---:|
| european_multiaction_cost_v1 | physical_only | 0.595285 [0.133100, 1.405715] | 30/186 | 23 | 26702 (343) |
| european_multiaction_cost_v1 | physical_only_harm2 | 0.356605 [0.062275, 0.887776] | 81/186 | 28 | 14480 (174) |
| european_multiaction_cost_v1 | physical_only_harm4 | 0.227536 [0.032097, 0.585635] | 106/186 | 37 | 7273 (106) |
| european_multiaction_cost_v1 | physical_neural | 0.553787 [0.142025, 1.289261] | 20/186 | 22 | 25478 (340) |
| european_multiaction_cost_v1 | physical_neural_harm2 | 0.344899 [0.066444, 0.852471] | 65/186 | 23 | 13505 (175) |
| european_multiaction_cost_v1 | physical_neural_harm4 | 0.221433 [0.033488, 0.555853] | 102/186 | 29 | 6815 (105) |
| european_multiaction_cost_v1 | neural_only | 0.331112 [0.179507, 0.533641] | 32/186 | 34 | 11541 (171) |
| european_multiaction_cost_v1 | neural_only_harm2 | 0.207028 [0.088048, 0.393169] | 47/186 | 50 | 6231 (106) |
| european_multiaction_cost_v1 | neural_only_harm4 | 0.123377 [0.042513, 0.245824] | 53/186 | 68 | 3158 (66) |
| european_multiaction_record_policy_v1 | physical_only | 0.000000 [0.000000, 0.000001] | 0/186 | 186 | 1 (0) |
| european_multiaction_record_policy_v1 | physical_neural | 0.000000 [0.000000, 0.000000] | 0/186 | 186 | 0 (0) |
| european_multiaction_record_policy_v1 | neural_only | 0.107591 [0.005193, 0.298005] | 14/186 | 158 | 5962 (80) |
| european_regression_deferral_v1 | physical_only | 7.229477 [4.289618, 10.000509] | 0/186 | 0 | 376642 (8019) |
| european_regression_deferral_v1 | physical_only_cost_gate | 0.479507 [0.116588, 1.108975] | 40/186 | 23 | 24209 (305) |
| european_regression_deferral_v1 | physical_neural | 8.018405 [5.042421, 10.822614] | 0/186 | 0 | 381203 (8144) |
| european_regression_deferral_v1 | physical_neural_cost_gate | 0.472543 [0.124128, 1.091729] | 30/186 | 23 | 23400 (309) |
| european_regression_deferral_v1 | neural_only | 7.338229 [4.798652, 9.756195] | 0/186 | 3 | 294153 (6170) |
| european_regression_deferral_v1 | neural_only_cost_gate | 0.288656 [0.159350, 0.447278] | 33/186 | 36 | 9984 (141) |
| european_multiaction_support_scale_v1 | physical_only | 0.053946 [0.013604, 0.115866] | 96/186 | 16 | 10459 (101) |
| european_multiaction_support_scale_v1 | physical_neural | 0.034796 [0.002944, 0.084866] | 92/186 | 15 | 8279 (77) |
| european_multiaction_support_scale_v1 | neural_only | 0.149501 [0.053194, 0.269304] | 56/186 | 20 | 17369 (164) |

## Paired Neural Increment and Missing Outcomes

Mixed minus physical-only, in percentage points. Completion bounds include
unknown selected outcomes; known-label gain alone is not robust utility.
Matched controls compare utility only, not safety.

| Study | Contrast | Point [conditional 95% CI] |
|---|---|---:|
| european_multiaction_cost_v1 | full_known_pp | -0.041498 [-0.124505, 0.025491] |
| european_multiaction_cost_v1 | full_lower_pp | -0.810703 [-2.385747, 0.005279] |
| european_multiaction_cost_v1 | full_upper_pp | 0.727708 [-0.021772, 2.171190] |
| european_multiaction_cost_v1 | matched_known_pp | -0.001607 [-0.033013, 0.026134] |
| european_multiaction_cost_v1 | matched_lower_pp | -0.464531 [-1.378516, 0.011413] |
| european_multiaction_record_policy_v1 | full_known_pp | -0.000000 [-0.000001, 0.000000] |
| european_multiaction_record_policy_v1 | full_lower_pp | -0.000000 [-0.000001, 0.000000] |
| european_multiaction_record_policy_v1 | full_upper_pp | -0.000000 [-0.000001, 0.000000] |
| european_multiaction_record_policy_v1 | matched_known_pp | 0.000000 [0.000000, 0.000000] |
| european_multiaction_record_policy_v1 | matched_lower_pp | 0.000000 [0.000000, 0.000000] |
| european_regression_deferral_v1 | full_known_pp | 0.788928 [-0.170983, 1.675128] |
| european_regression_deferral_v1 | full_lower_pp | -4.463329 [-10.643677, -1.085987] |
| european_regression_deferral_v1 | full_upper_pp | 6.041185 [2.750807, 11.258337] |
| european_regression_deferral_v1 | matched_known_pp | 0.893883 [0.162533, 1.689813] |
| european_regression_deferral_v1 | matched_lower_pp | -3.156553 [-7.664351, -0.605739] |
| european_multiaction_support_scale_v1 | full_known_pp | -0.019150 [-0.037903, -0.004219] |
| european_multiaction_support_scale_v1 | full_lower_pp | -0.043270 [-0.104130, -0.005320] |
| european_multiaction_support_scale_v1 | full_upper_pp | 0.004970 [-0.026442, 0.048551] |
| european_multiaction_support_scale_v1 | matched_known_pp | 0.003608 [0.000222, 0.008673] |
| european_multiaction_support_scale_v1 | matched_lower_pp | 0.000831 [-0.000661, 0.002283] |

## Training and Failure Boundaries

| Study | Fits | Primary optimizer updates | Seconds | Peak RSS bytes |
|---|---:|---:|---:|---:|
| european_multiaction_cost_v1 | 54 | 108000 | 210.034352 | 10093199360 |
| european_multiaction_record_policy_v1 | 54 | 27000 | 984.886601 | 5949882368 |
| european_regression_deferral_v1 | 54 | 108000 | 200.797506 | 11402346496 |
| european_multiaction_support_scale_v1 | 54 | 27000 | 992.385098 | 5794103296 |

## Final Training Monitors

These monitors precede the last optimizer update; endpoint behavior is in
the fixed readout above. Support-unit magnitudes are not compared across
the old/new units, only invariant constraint signs and action counts.

| Study | Bank | Zero-intervention fits | All/easy/support violations | Fully supported monitor views |
|---|---|---:|---:|---:|
| european_multiaction_record_policy_v1 | physical_only | 18/18 | 0/0/444 | 0/444 |
| european_multiaction_record_policy_v1 | physical_neural | 18/18 | 0/0/444 | 0/444 |
| european_multiaction_record_policy_v1 | neural_only | 15/18 | 11/40/375 | 29/444 |
| european_multiaction_support_scale_v1 | physical_only | 0/18 | 107/165/12 | 248/444 |
| european_multiaction_support_scale_v1 | physical_neural | 0/18 | 122/182/12 | 228/444 |
| european_multiaction_support_scale_v1 | neural_only | 0/18 | 182/240/33 | 160/444 |

All action-bank arms retain their preregistered seeds and complete roster.
The published deferral loss is an application adapter, not a reproduction of
its original experiments or bounded-loss theorem verification. Its cost-gated
variant is a fixed control, not a calibrated deployment policy.

The support-unit repair changes only the numerical unit of the third fitting
constraint; all/easy2% budgets and the support feasible set stay unchanged.
Any success or failure must be interpreted alongside full completion bounds,
every-record risk, the physical-only contrast, and fitting behavior. A zero
intervention count is not a passed selected-risk gate. Published loss values
and primal-dual objectives are different quantities, not cross-method scores.

These are frozen-forecaster routing studies, not a new neural trajectory lift
or scene-joint optimizer. Selection, calibration and confirmation remain closed.
M3W remains a2.5D trajectory scaffold; pixel/raw-frame results are not metric
or effective-second claims. No deployment, Stage5C execution or SMC is enabled.
