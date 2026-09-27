# Source-Separated Calibration: Matched Readout

Fresh_run: causal inference, empirical two-source calibration and held-source scoring.
Cached_verified: frozen forecast banks and risk/utility/easy checkpoints. New training updates: 0.
All results use development-exposed source localities, not independent confirmation.

## Primary Comparison

Signed-score calibrated-supported neural versus equally protected damping ADE gain (%): **-0.6699 [-0.9742, -0.3964]**.

## All Prespecified Controls

| Candidate | Policy | ADE gain vs CV % | Easy gain % | Hard gain % | Switch % | Risk-violating views | Worst view easy gain % |
|---|---|---:|---:|---:|---:|---:|---:|
| dimensionless | excess_calibrated | 0.1796 [0.1348, 0.2318] | 3.4552 [2.4727, 4.4666] | 0.0530 [0.0209, 0.0904] | 9.8603 [7.7625, 11.8403] | 6 | 0.0000 |
| dimensionless | excess_calibrated_supported | 0.1666 [0.1206, 0.2205] | 3.4125 [2.4281, 4.4374] | 0.0478 [0.0170, 0.0853] | 9.7901 [7.6956, 11.7816] | 9 | 0.0000 |
| dimensionless | excess_guarded | 0.2825 [0.2081, 0.3593] | 4.3140 [2.8246, 5.9569] | 0.1022 [0.0402, 0.1699] | 14.0877 [11.1673, 17.1543] | 75 | -2.1993 |
| dimensionless | excess_supported | 0.2327 [0.1734, 0.3083] | 4.2662 [2.7796, 5.9220] | 0.0634 [0.0274, 0.1009] | 13.9630 [11.0341, 17.0534] | 81 | -2.1993 |
| dimensionless | legacy | 0.3342 [0.2547, 0.4224] | 3.9961 [2.7207, 5.4116] | 0.1549 [0.0693, 0.2463] | 13.2518 [9.9621, 16.5127] | 87 | -0.6108 |
| dimensionless | moments_calibrated | 0.1872 [0.1096, 0.2880] | 2.5341 [1.4082, 3.8723] | 0.0502 [0.0158, 0.0965] | 7.9198 [5.0015, 11.0996] | 20 | -0.1444 |
| dimensionless | moments_calibrated_supported | 0.1319 [0.0888, 0.1790] | 2.3007 [1.3600, 3.4067] | 0.0437 [0.0122, 0.0897] | 7.2501 [4.7447, 9.9099] | 19 | 0.0000 |
| dimensionless | moments_guarded | 0.3058 [0.2270, 0.3952] | 4.0727 [2.8542, 5.4491] | 0.1202 [0.0460, 0.2034] | 13.2179 [10.0012, 16.4070] | 81 | -0.3191 |
| dimensionless | moments_supported | 0.2293 [0.1789, 0.2839] | 4.0367 [2.8041, 5.4384] | 0.0859 [0.0363, 0.1389] | 13.0898 [9.8386, 16.2900] | 96 | -0.2450 |
| damped | excess_calibrated | 0.8335 [0.5567, 1.1210] | 3.8407 [2.9981, 4.5556] | 0.4446 [0.2085, 0.6871] | 35.6996 [29.1374, 41.9935] | 0 | 0.1214 |
| damped | excess_calibrated_supported | 0.8259 [0.5478, 1.1164] | 3.8347 [2.9877, 4.5527] | 0.4423 [0.2067, 0.6841] | 35.6017 [28.9792, 41.9063] | 0 | 0.1214 |
| damped | excess_guarded | 0.8335 [0.5567, 1.1210] | 3.8407 [2.9981, 4.5556] | 0.4446 [0.2085, 0.6871] | 35.6996 [29.1374, 41.9935] | 0 | 0.1214 |
| damped | excess_supported | 0.8259 [0.5478, 1.1164] | 3.8347 [2.9877, 4.5527] | 0.4423 [0.2067, 0.6841] | 35.6017 [28.9792, 41.9063] | 0 | 0.1214 |
| damped | legacy | 0.6994 [0.4878, 0.9297] | 3.4998 [2.7230, 4.1773] | 0.3933 [0.1914, 0.5996] | 30.8699 [25.2849, 36.5819] | 0 | 0.2456 |
| damped | moments_calibrated | 0.6931 [0.4843, 0.9212] | 3.5903 [2.8025, 4.2682] | 0.3643 [0.1792, 0.5552] | 31.7798 [25.9078, 37.5917] | 0 | 0.2623 |
| damped | moments_calibrated_supported | 0.6858 [0.4738, 0.9160] | 3.5848 [2.7973, 4.2627] | 0.3618 [0.1771, 0.5505] | 31.6886 [25.8060, 37.5473] | 0 | 0.2623 |
| damped | moments_guarded | 0.6931 [0.4843, 0.9212] | 3.5903 [2.8025, 4.2682] | 0.3643 [0.1792, 0.5552] | 31.7798 [25.9078, 37.5917] | 0 | 0.2623 |
| damped | moments_supported | 0.6858 [0.4738, 0.9160] | 3.5848 [2.7973, 4.2627] | 0.3618 [0.1771, 0.5505] | 31.6886 [25.8060, 37.5473] | 0 | 0.2623 |

## Neural Versus Matched Damping

| Policy | Paired ADE gain % |
|---|---:|
| excess_calibrated | -0.6644 [-0.9675, -0.3959] |
| excess_calibrated_supported | -0.6699 [-0.9742, -0.3964] |
| excess_guarded | -0.5609 [-0.8504, -0.2956] |
| excess_supported | -0.6034 [-0.8927, -0.3345] |
| legacy | -0.3707 [-0.6136, -0.1252] |
| moments_calibrated | -0.5127 [-0.7834, -0.2367] |
| moments_calibrated_supported | -0.5608 [-0.7995, -0.3392] |
| moments_guarded | -0.3928 [-0.6301, -0.1541] |
| moments_supported | -0.4623 [-0.6710, -0.2650] |

## Primary By Seed

| Seed | Paired ADE gain % |
|---|---:|
| 17 | -0.5290 [-0.8052, -0.2813] |
| 29 | -0.7494 [-1.0671, -0.4680] |
| 43 | -0.7312 [-1.0608, -0.4217] |

## Checks and Claim Limits

- primary_neural_vs_damping_lower_CI_positive: False
- neural_gain_vs_CV_positive: True
- every_held_view_easy_preserved: True
- every_locality_nonzero_coverage: True
- no_zero_reference_harm: True
- no_observed_risk_violation: False
- exploratory_complete_screen_pass: False
- calibration_certificate: False
- independent_confirmation: False
- deployment_changed: False
- stage5c_executed: False
- smc_enabled: False

Intervals resample twelve locality means after averaging dependent producer/seed/calibration views.
Each candidate/policy has216 held locality views. Overlapping windows are not independent samples.
Positive-harm/reference is separate from net ADE or easy degradation. Empty selected ratios remain undefined.
The complete static guard is present, but two-source empirical calibration is not a conformal certificate.
Both objective families remain controls; no held-result model/threshold winner was chosen.
No new independent source, scene-joint benefit or deployment promotion follows.
Silver image-local obs8/pred12 at raw-frame stride12; not Stage37t50, metric, seconds, human gold,
physical safety, true3D or foundation evidence. Stage5C/SMC remain disabled.
