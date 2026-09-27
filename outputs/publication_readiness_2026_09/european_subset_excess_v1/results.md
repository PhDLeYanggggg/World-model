# Anchored Subset Risk Training

## Evidence Role

fresh_run: 216 new risk heads, 432,000 updates and outcome readout. cached_verified: predictors, floor, utility and previous controls.
Twelve opened source-training development localities; 4producer/4controller/2risk-fit/2held roles, three forecasting seeds.
Independent selection/calibration/confirmation remain closed. No held checkpoint or threshold selection. No deployment change.
Image-local detector silver, obs8/pred12 raw-frame stride12. No metric, seconds, human-gold, physical-safety, true3D or foundation claim. No Stage5C or SMC.

## Registered Development Contrasts

| Contrast | ADE gain % | Selected harm reduction pp | All-reference harm reduction pp | Intervention difference pp |
|---|---:|---:|---:|---:|
| subset_aggregate_joint_vs_cached_pointwise_joint | -0.0120 [-0.0274, 0.0033] | undefined | -0.0001 [-0.0059, 0.0056] | -0.3522 [-0.4578, -0.2288] |
| subset_aggregate_joint_vs_subset_pointwise_joint | 0.0035 [-0.0115, 0.0177] | undefined | -0.0109 [-0.0201, -0.0034] | -0.1356 [-0.2380, -0.0305] |
| subset_aggregate_rank_vs_cached_pointwise_rank | 0.0046 [0.0001, 0.0097] | undefined | -0.0017 [-0.0044, 0.0011] | 0.0000 [0.0000, 0.0000] |
| subset_aggregate_rank_vs_subset_pointwise_rank | 0.0007 [-0.0087, 0.0105] | undefined | -0.0023 [-0.0047, -0.0001] | 0.0000 [0.0000, 0.0000] |

Rank contrasts use the same original per-query intervention count. Joint contrasts do not match intervention counts.
Ten inherited zero-action views leave selected-risk undefined. Fixed-denominator total harm is zero under abstention, not a substitute2%risk certificate. The old primary remains incomplete.

## Policy Results

| Policy | ADE/floor % | Hard/floor % | Easy/CV % | FDE/floor % | Intervention % | Selected harm % | All-reference harm % |
|---|---:|---:|---:|---:|---:|---:|---:|
| cached_pointwise_joint | 0.5201 [0.2755, 0.7977] | 0.5556 [0.2714, 0.8751] | 5.3864 [4.1059, 6.5819] | 0.7419 [0.4000, 1.1376] | 8.1507 [6.2026, 10.0986] | undefined | 0.1672 [0.0786, 0.2690] |
| cached_pointwise_rank | 0.2670 [0.1750, 0.3679] | 0.2261 [0.1129, 0.3582] | 5.6505 [4.4212, 6.7865] | 0.3848 [0.2641, 0.5204] | 8.0235 [6.4457, 9.5834] | undefined | 0.0888 [0.0388, 0.1476] |
| cached_query_joint | 0.3706 [0.2064, 0.5536] | 0.3825 [0.1773, 0.6045] | 5.2967 [4.0356, 6.4873] | 0.5269 [0.3025, 0.7737] | 7.4297 [5.5430, 9.4301] | undefined | 0.1432 [0.0652, 0.2323] |
| cached_query_rank | 0.2759 [0.1694, 0.3967] | 0.2359 [0.1069, 0.3899] | 5.5976 [4.3750, 6.7345] | 0.3951 [0.2544, 0.5581] | 8.0235 [6.4457, 9.5834] | undefined | 0.0946 [0.0444, 0.1534] |
| floor | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 3.8347 [2.9848, 4.5433] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | undefined | 0.0000 [0.0000, 0.0000] |
| parent_joint | 0.4882 [0.2998, 0.7062] | 0.4996 [0.2634, 0.7590] | 5.3507 [4.1625, 6.4835] | 0.6946 [0.4400, 0.9963] | 8.0235 [6.4457, 9.5834] | undefined | 0.1504 [0.0697, 0.2417] |
| subset_aggregate_independent | 0.2688 [0.1490, 0.4050] | 0.2544 [0.1126, 0.4187] | 5.6300 [4.3134, 6.8488] | 0.3860 [0.2184, 0.5798] | 7.7985 [5.8439, 9.7245] | undefined | 0.0929 [0.0369, 0.1613] |
| subset_aggregate_joint | 0.5079 [0.2631, 0.7902] | 0.5580 [0.2732, 0.8755] | 5.3252 [4.0647, 6.4996] | 0.7251 [0.3816, 1.1224] | 7.7985 [5.8439, 9.7245] | undefined | 0.1673 [0.0773, 0.2673] |
| subset_aggregate_rank | 0.2714 [0.1801, 0.3719] | 0.2343 [0.1183, 0.3693] | 5.6465 [4.4198, 6.7765] | 0.3931 [0.2726, 0.5297] | 8.0235 [6.4457, 9.5834] | undefined | 0.0904 [0.0394, 0.1518] |
| subset_pointwise_independent | 0.2683 [0.1483, 0.4045] | 0.2424 [0.1016, 0.4027] | 5.6888 [4.3691, 6.9114] | 0.3845 [0.2174, 0.5779] | 7.9341 [5.9977, 9.8303] | undefined | 0.0848 [0.0326, 0.1494] |
| subset_pointwise_joint | 0.5049 [0.2652, 0.7802] | 0.5384 [0.2617, 0.8495] | 5.3799 [4.1043, 6.5721] | 0.7182 [0.3817, 1.1079] | 7.9341 [5.9977, 9.8303] | undefined | 0.1564 [0.0724, 0.2530] |
| subset_pointwise_rank | 0.2707 [0.1764, 0.3756] | 0.2315 [0.1128, 0.3668] | 5.6494 [4.4172, 6.7841] | 0.3896 [0.2660, 0.5313] | 8.0235 [6.4457, 9.5834] | undefined | 0.0881 [0.0386, 0.1472] |

## Coverage, Risk and Tail

| Policy | Violating /216 | Undefined | Abstaining | Worst easy gain % | P95/floor | Unknown interventions/view |
|---|---:|---:|---:|---:|---:|---:|
| cached_pointwise_joint | 89 | 14 | 14 | 0.188564 | 0.9958 [0.9931, 0.9982] | 29.9954 [5.3468, 63.4216] |
| cached_pointwise_rank | 89 | 10 | 10 | 0.346417 | 0.9986 [0.9978, 0.9993] | 25.0509 [5.0926, 54.0611] |
| cached_query_joint | 98 | 12 | 12 | 0.145638 | 0.9973 [0.9957, 0.9988] | 29.9306 [4.6155, 66.2547] |
| cached_query_rank | 95 | 10 | 10 | 0.361736 | 0.9983 [0.9970, 0.9993] | 25.3981 [5.1894, 54.7177] |
| floor | 0 | 216 | 216 | 0.121432 | 1.0000 [1.0000, 1.0000] | 0.0000 [0.0000, 0.0000] |
| parent_joint | 99 | 10 | 10 | 0.371606 | 0.9963 [0.9941, 0.9982] | 28.0787 [5.3191, 59.1446] |
| subset_aggregate_independent | 65 | 16 | 16 | 0.176646 | 0.9984 [0.9973, 0.9994] | 24.8657 [4.8102, 53.4125] |
| subset_aggregate_joint | 84 | 16 | 16 | 0.176646 | 0.9958 [0.9930, 0.9981] | 28.8843 [5.3237, 61.2506] |
| subset_aggregate_rank | 88 | 10 | 10 | 0.346417 | 0.9985 [0.9977, 0.9993] | 25.0463 [5.1574, 53.8738] |
| subset_pointwise_independent | 68 | 13 | 13 | 0.237213 | 0.9984 [0.9974, 0.9994] | 24.0602 [4.9212, 51.4978] |
| subset_pointwise_joint | 82 | 13 | 13 | 0.237213 | 0.9959 [0.9932, 0.9982] | 28.0093 [5.2499, 59.1485] |
| subset_pointwise_rank | 88 | 10 | 10 | 0.372458 | 0.9985 [0.9977, 0.9993] | 25.0046 [5.1338, 53.8388] |

## Held Risk Prediction

| Arm | Metric | Estimate and locality95%CI |
|---|---|---:|
| subset_aggregate | pointwise_all_MSE | 0.1803 [0.1174, 0.2449] |
| subset_aggregate | pointwise_easy_MSE | 0.0030 [0.0018, 0.0044] |
| subset_aggregate | query_all_MSE | 0.0770 [0.0467, 0.1160] |
| subset_aggregate | query_easy_MSE | 0.0019 [0.0007, 0.0033] |
| subset_aggregate | queries | 3417.9167 [1055.9042, 6816.3813] |
| subset_aggregate | singleton_fraction | 0.2996 [0.1788, 0.4312] |
| subset_aggregate | controller_admission_all_MSE | 0.0006 [0.0003, 0.0011] |
| subset_aggregate | controller_admission_easy_MSE | 0.0001 [0.0000, 0.0002] |
| subset_aggregate | controller_admission_known_nonempty_fraction | 0.3774 [0.2625, 0.5107] |
| subset_aggregate | low_disagreement_half_all_MSE | 0.0526 [0.0254, 0.0888] |
| subset_aggregate | low_disagreement_half_easy_MSE | 0.0019 [0.0007, 0.0034] |
| subset_aggregate | low_disagreement_half_known_nonempty_fraction | 0.9977 [0.9968, 0.9986] |
| subset_aggregate | high_disagreement_half_all_MSE | 0.2107 [0.1371, 0.2866] |
| subset_aggregate | high_disagreement_half_easy_MSE | 0.0023 [0.0014, 0.0033] |
| subset_aggregate | high_disagreement_half_known_nonempty_fraction | 0.6992 [0.5679, 0.8196] |
| subset_pointwise | pointwise_all_MSE | 0.1805 [0.1172, 0.2458] |
| subset_pointwise | pointwise_easy_MSE | 0.0030 [0.0018, 0.0043] |
| subset_pointwise | query_all_MSE | 0.0774 [0.0469, 0.1173] |
| subset_pointwise | query_easy_MSE | 0.0019 [0.0007, 0.0032] |
| subset_pointwise | queries | 3417.9167 [1055.9042, 6816.3813] |
| subset_pointwise | singleton_fraction | 0.2996 [0.1788, 0.4312] |
| subset_pointwise | controller_admission_all_MSE | 0.0007 [0.0003, 0.0011] |
| subset_pointwise | controller_admission_easy_MSE | 0.0001 [0.0000, 0.0002] |
| subset_pointwise | controller_admission_known_nonempty_fraction | 0.3774 [0.2625, 0.5107] |
| subset_pointwise | low_disagreement_half_all_MSE | 0.0533 [0.0256, 0.0900] |
| subset_pointwise | low_disagreement_half_easy_MSE | 0.0019 [0.0007, 0.0033] |
| subset_pointwise | low_disagreement_half_known_nonempty_fraction | 0.9977 [0.9968, 0.9986] |
| subset_pointwise | high_disagreement_half_all_MSE | 0.2098 [0.1368, 0.2860] |
| subset_pointwise | high_disagreement_half_easy_MSE | 0.0023 [0.0014, 0.0033] |
| subset_pointwise | high_disagreement_half_known_nonempty_fraction | 0.6992 [0.5679, 0.8196] |

Held subset losses average nonempty known subsets; fitting loss includes empty subsets as zero. Their raw magnitudes are not directly comparable.
Subset coverage is built from all causal rows before unknown labels are excluded from supervised loss. Unknown rows stay in inference/action counts.
Controller admission is a fixed past-only proxy from a separately fitted source group, not an exact deployed floor-relative mask.
The individual anchor is shared. Within-subset mean-square error versus square-of-mean error is the only matched objective difference.
The latter is algebraically no larger for a fixed model. That inequality alone is not learned improvement; compare models on the same held metric.

## Fitting and Gates

- subset_pointwise: {'heads': 108, 'updates': 216000, 'training_queries': 738270, 'singleton_queries': 167796, 'loss_declined_heads': 108, 'unknown_draws': 0}
- subset_aggregate: {'heads': 108, 'updates': 216000, 'training_queries': 738270, 'singleton_queries': 167796, 'loss_declined_heads': 108, 'unknown_draws': 0}

- equal_count_ADE_advantage: False
- equal_count_all_reference_harm_reduction: False
- ranks_count_matched: True
- joint_ADE_advantage: False
- every_view_defined_selected_risk_within_2percent: False
- every_view_easy_preserved: True
- no_zero_CV_harm: True
- exploratory_joint_screen_pass: False
- formal_primary_replaced: False
- independent_confirmation: False
- calibration_certificate: False
- deployment_changed: False
- stage5c_executed: False
- smc_enabled: False

Fitting-query totals repeat data across heads and are not independent samples. CIs use3000locality draws after averaging dependent views; they are development CIs.
Four score outputs are not separately identified/calibrated expected moments. No new collision or physical-safety model was trained.
Retained query cohorts are not a completeness guarantee for every visible agent; missing future labels are not assumed missing at random.
Full legacy tests and cold raw reconstruction are not_run; scoped verification does not substitute independent generalization.
