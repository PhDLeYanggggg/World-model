# Cost Features and Decision Failures: Six Controlled Studies

This new revision preserves the earlier four-study export and manuscript.
Training, frozen endpoints and exact readouts are source evidence; this
export reuses their utility intervals and computes descriptive paired moment
MSE intervals from verified aggregates. It does not train or select a model.

## Primary Inner-Held Results

| Study | Bank | Known ADE gain % | Conditional95%CI | Required risk views supported | Undefined easy |
|---|---|---:|---:|---:|---:|
| european_multiaction_cost_v1 | physical_only | 0.595285 | [0.133100,1.405715] | 30/186 | 23 |
| european_multiaction_cost_v1 | physical_neural | 0.553787 | [0.142025,1.289261] | 20/186 | 22 |
| european_multiaction_cost_v1 | neural_only | 0.331112 | [0.179507,0.533641] | 32/186 | 34 |
| european_multiaction_record_policy_v1 | physical_only | 0.000000 | [0.000000,0.000001] | 0/186 | 186 |
| european_multiaction_record_policy_v1 | physical_neural | 0.000000 | [0.000000,0.000000] | 0/186 | 186 |
| european_multiaction_record_policy_v1 | neural_only | 0.107591 | [0.005193,0.298005] | 14/186 | 158 |
| european_regression_deferral_v1 | physical_only | 7.229477 | [4.289618,10.000509] | 0/186 | 0 |
| european_regression_deferral_v1 | physical_neural | 8.018405 | [5.042421,10.822614] | 0/186 | 0 |
| european_regression_deferral_v1 | neural_only | 7.338229 | [4.798652,9.756195] | 0/186 | 3 |
| european_multiaction_support_scale_v1 | physical_only | 0.053946 | [0.013604,0.115866] | 96/186 | 16 |
| european_multiaction_support_scale_v1 | physical_neural | 0.034796 | [0.002944,0.084866] | 92/186 | 15 |
| european_multiaction_support_scale_v1 | neural_only | 0.149501 | [0.053194,0.269304] | 56/186 | 20 |
| european_shared_reference_cost_v1 | physical_only | 0.553001 | [0.128801,1.286377] | 30/186 | 25 |
| european_shared_reference_cost_v1 | physical_neural | 0.542017 | [0.141372,1.246147] | 21/186 | 20 |
| european_shared_reference_cost_v1 | neural_only | 0.276137 | [0.131230,0.487250] | 31/186 | 35 |
| european_static_scene_cost_v1 | physical_only | 1.076598 | [0.258362,2.522915] | 24/186 | 17 |
| european_static_scene_cost_v1 | physical_neural | 0.982769 | [0.242283,2.301831] | 23/186 | 20 |
| european_static_scene_cost_v1 | neural_only | 0.433381 | [0.206401,0.696722] | 31/186 | 40 |

## Paired Neural Contribution

Mixed minus physical-only,percentage points. Known-label intervals cannot
substitute for completion bounds or the every-record all/easy2%contract.

| Study | Known difference [95%CI] | Completion-lower difference [95%CI] |
|---|---:|---:|
| european_multiaction_cost_v1 | -0.041498 [-0.124505,0.025491] | -0.810703 [-2.385747,0.005279] |
| european_multiaction_record_policy_v1 | -0.000000 [-0.000001,0.000000] | -0.000000 [-0.000001,0.000000] |
| european_regression_deferral_v1 | 0.788928 [-0.170983,1.675128] | -4.463329 [-10.643677,-1.085987] |
| european_multiaction_support_scale_v1 | -0.019150 [-0.037903,-0.004219] | -0.043270 [-0.104130,-0.005320] |
| european_shared_reference_cost_v1 | -0.010984 [-0.054111,0.028694] | -0.809107 [-2.386719,0.001198] |
| european_static_scene_cost_v1 | -0.093829 [-0.217875,-0.003382] | -1.195683 [-3.282407,-0.065640] |

## Does Better Moment Prediction Explain Better Decisions?

New minus original normalized MSE;negative is lower squared error.
These are fresh3,000-locality bootstrap calculations on known-label
aggregates,not independent confirmation or a familywise claim.

| Ablation | Scope | Mixed cost | MSE change [conditional95%CI] |
|---|---|---|---:|
| european_shared_reference_cost_v1 | fit | benefit | 0.000639 [-0.000430,0.001829] |
| european_shared_reference_cost_v1 | fit | harm | -0.000568 [-0.013924,0.009293] |
| european_shared_reference_cost_v1 | fit | reference | -0.047954 [-0.066402,-0.028348] |
| european_shared_reference_cost_v1 | fit | easy_reference | -0.000073 [-0.000181,-0.000012] |
| european_shared_reference_cost_v1 | fit | easy_harm | 0.000276 [-0.000091,0.000808] |
| european_shared_reference_cost_v1 | inner_held | benefit | -0.000559 [-0.007103,0.004528] |
| european_shared_reference_cost_v1 | inner_held | harm | -0.008473 [-0.026645,0.009147] |
| european_shared_reference_cost_v1 | inner_held | reference | -0.019930 [-0.059330,0.016005] |
| european_shared_reference_cost_v1 | inner_held | easy_reference | 0.000082 [-0.000080,0.000322] |
| european_shared_reference_cost_v1 | inner_held | easy_harm | 0.001134 [0.000195,0.002650] |
| european_static_scene_cost_v1 | fit | benefit | -0.005442 [-0.008586,-0.002022] |
| european_static_scene_cost_v1 | fit | harm | -0.059359 [-0.091775,-0.023008] |
| european_static_scene_cost_v1 | fit | reference | -0.091613 [-0.118358,-0.061239] |
| european_static_scene_cost_v1 | fit | easy_reference | -0.000139 [-0.000278,-0.000017] |
| european_static_scene_cost_v1 | fit | easy_harm | -0.003084 [-0.006764,-0.000648] |
| european_static_scene_cost_v1 | inner_held | benefit | 0.000189 [-0.004746,0.005832] |
| european_static_scene_cost_v1 | inner_held | harm | -0.106608 [-0.173896,-0.047766] |
| european_static_scene_cost_v1 | inner_held | reference | 0.006804 [-0.073022,0.103460] |
| european_static_scene_cost_v1 | inner_held | easy_reference | -0.000809 [-0.002125,0.000066] |
| european_static_scene_cost_v1 | inner_held | easy_harm | -0.007039 [-0.016221,0.000124] |

## Actual Training, Not a Fresh Forecaster

| Study | New heads | Primary updates | Seconds |
|---|---:|---:|---:|
| european_multiaction_cost_v1 | 54 | 108000 | 210.034 |
| european_multiaction_record_policy_v1 | 54 | 27000 | 984.887 |
| european_regression_deferral_v1 | 54 | 108000 | 200.798 |
| european_multiaction_support_scale_v1 | 54 | 27000 | 992.385 |
| european_shared_reference_cost_v1 | 54 | 108000 | 366.152 |
| european_static_scene_cost_v1 | 54 | 108000 | 339.606 |

## Claim Boundary

The trajectory producer is fixed at seed43;three seeds refer to controller
heads,not three newly trained full forecasting pipelines. Repeated context/
head views are not independent recordings. SOURCE inner-held was exposed.
No whole-pipeline OOF or independent selection/calibration/confirmation is claimed.
Empty risk is not_estimable;count-matched controls measure utility only.
MSE reduction alone is neither neural utility nor a safety certificate.
Raw pixel/frame webcam detector tracks are not metric/seconds or human gold.
No deployment promotion,Stage5C execution,SMC or submission readiness.
