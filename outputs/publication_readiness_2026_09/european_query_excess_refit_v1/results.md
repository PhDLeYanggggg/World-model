# Matched Query-Excess Training Results

## Material Passport

Fresh paired risk-head training and outcome evaluation; cached_verified forecasters, floor and utility.
108 paired groups, 216 new heads, 432,000 updates. Twelve opened development localities; three forecasting seeds.
Independent selection/calibration/confirmation remain closed. No outcome-selected checkpoints or thresholds.

## Registered Primary

Pre-readout disclosure: ten zero-action parent views force undefined selected-risk ratios in both matched-count arms. The full-roster primary is structurally incomplete, not a successful primary test. See primary_feasibility_addendum.md.
Query versus pointwise ranking uses the same frozen-parent count in every current query. These rank arms are diagnostics, not risk-certified policies.
Selected positive-harm reduction (percentage points): **undefined**.
Equal-count ADE gain (%): **0.0090 [-0.0140, 0.0418]**.

| Policy | ADE / floor gain % | Hard / floor gain % | Easy / CV gain % | FDE / floor gain % | Intervention % | Selected harm % |
|---|---:|---:|---:|---:|---:|---:|
| floor | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 3.8347 [2.9848, 4.5433] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | undefined |
| parent_independent | 0.2594 [0.1632, 0.3687] | 0.1923 [0.0801, 0.3321] | 5.6406 [4.4125, 6.7641] | 0.3719 [0.2449, 0.5184] | 8.0235 [6.4457, 9.5834] | undefined |
| parent_joint | 0.4882 [0.2998, 0.7062] | 0.4996 [0.2634, 0.7590] | 5.3507 [4.1625, 6.4835] | 0.6946 [0.4400, 0.9963] | 8.0235 [6.4457, 9.5834] | undefined |
| pointwise_independent | 0.2710 [0.1527, 0.4045] | 0.2418 [0.1039, 0.4004] | 5.7227 [4.4001, 6.9557] | 0.3874 [0.2248, 0.5753] | 8.1507 [6.2026, 10.0986] | undefined |
| pointwise_joint | 0.5201 [0.2755, 0.7977] | 0.5556 [0.2714, 0.8751] | 5.3864 [4.1059, 6.5819] | 0.7419 [0.4000, 1.1376] | 8.1507 [6.2026, 10.0986] | undefined |
| pointwise_rank | 0.2670 [0.1750, 0.3679] | 0.2261 [0.1129, 0.3582] | 5.6505 [4.4212, 6.7865] | 0.3848 [0.2641, 0.5204] | 8.0235 [6.4457, 9.5834] | undefined |
| query_independent | 0.2120 [0.1192, 0.3182] | 0.1772 [0.0687, 0.3000] | 5.5595 [4.2373, 6.7777] | 0.3050 [0.1781, 0.4522] | 7.4297 [5.5430, 9.4301] | undefined |
| query_joint | 0.3706 [0.2064, 0.5536] | 0.3825 [0.1773, 0.6045] | 5.2967 [4.0356, 6.4873] | 0.5269 [0.3025, 0.7737] | 7.4297 [5.5430, 9.4301] | undefined |
| query_rank | 0.2759 [0.1694, 0.3967] | 0.2359 [0.1069, 0.3899] | 5.5976 [4.3750, 6.7345] | 0.3951 [0.2544, 0.5581] | 8.0235 [6.4457, 9.5834] | undefined |
| utility_topk_parent | 1.6807 [1.1117, 2.2587] | 2.4280 [1.7055, 3.1192] | 3.7559 [2.9407, 4.5040] | 2.4573 [1.6717, 3.2602] | 8.0235 [6.4457, 9.5834] | undefined |

## Paired Contrasts

| Contrast | ADE gain % | Harm reduction pp | Intervention difference pp |
|---|---:|---:|---:|
| query_joint_vs_parent_joint | -0.1202 [-0.2279, -0.0386] | undefined | -0.5938 [-1.4311, 0.2099] |
| query_joint_vs_pointwise_joint | -0.1547 [-0.3057, -0.0465] | undefined | -0.7210 [-1.2933, -0.2062] |
| query_rank_vs_pointwise_rank | 0.0090 [-0.0140, 0.0418] | undefined | 0.0000 [0.0000, 0.0000] |

The joint-policy contrasts use the same nominal predicted budget, not matched intervention counts. Only the registered ranking contrast is count matched.

## Risk, Missing Labels and Tail

| Policy | Violating views /216 | Undefined views | Worst easy gain % | P95 ratio / floor | Unknown actions per view |
|---|---:|---:|---:|---:|---:|
| floor | 0 | 216 | 0.121432 | 1.0000 [1.0000, 1.0000] | 0.0000 [0.0000, 0.0000] |
| parent_independent | 82 | 10 | 0.371606 | 0.9988 [0.9979, 0.9996] | 23.4954 [5.1706, 49.8024] |
| parent_joint | 99 | 10 | 0.371606 | 0.9963 [0.9941, 0.9982] | 28.0787 [5.3191, 59.1446] |
| pointwise_independent | 73 | 14 | 0.188564 | 0.9984 [0.9973, 0.9994] | 25.5972 [4.9943, 54.8434] |
| pointwise_joint | 89 | 14 | 0.188564 | 0.9958 [0.9931, 0.9982] | 29.9954 [5.3468, 63.4216] |
| pointwise_rank | 89 | 10 | 0.346417 | 0.9986 [0.9978, 0.9993] | 25.0509 [5.0926, 54.0611] |
| query_independent | 82 | 12 | 0.145638 | 0.9990 [0.9979, 1.0000] | 25.6111 [4.3330, 56.9935] |
| query_joint | 98 | 12 | 0.145638 | 0.9973 [0.9957, 0.9988] | 29.9306 [4.6155, 66.2547] |
| query_rank | 95 | 10 | 0.361736 | 0.9983 [0.9970, 0.9993] | 25.3981 [5.1894, 54.7177] |
| utility_topk_parent | 186 | 11 | -3.279719 | 0.9847 [0.9778, 0.9912] | 55.3611 [10.0814, 120.7906] |

## Held Query Prediction

| Objective | Pointwise all MSE | Pointwise easy MSE | Query all MSE | Query easy MSE | Singleton fraction |
|---|---:|---:|---:|---:|---:|
| pointwise | 0.1806 [0.1173, 0.2459] | 0.0030 [0.0018, 0.0044] | 0.0774 [0.0467, 0.1171] | 0.0019 [0.0007, 0.0033] | 0.2996 [0.1788, 0.4312] |
| query | 0.1847 [0.1207, 0.2511] | 0.0031 [0.0018, 0.0046] | 0.0784 [0.0472, 0.1172] | 0.0020 [0.0007, 0.0034] | 0.2996 [0.1788, 0.4312] |

Both objectives are evaluated on both losses. Aggregate MSE is algebraically no larger than individual MSE on a fixed model; that inequality is not a learned improvement. Compare models within the same metric.

## Training and Gates

- pointwise: {'heads': 108, 'updates': 216000, 'training_queries': 738270, 'singleton_queries': 167796, 'loss_declined_heads': 108, 'unknown_draws': 0}
- query: {'heads': 108, 'updates': 216000, 'training_queries': 738270, 'singleton_queries': 167796, 'loss_declined_heads': 104, 'unknown_draws': 0}

- primary_equal_count_harm_reduction: False
- same_count_ADE_advantage: False
- all_rank_counts_matched: True
- joint_ADE_advantage: False
- floor_ADE_advantage: True
- every_view_defined_risk_within_budget: False
- every_view_easy_preserved: True
- no_zero_CV_harm: True
- nonempty_localities: True
- exploratory_joint_screen_pass: False
- independent_confirmation: False
- calibration_certificate: False
- deployment_changed: False
- stage5c_executed: False
- smc_enabled: False

Training-query totals sum repeated fitting-query occurrences across heads; they are not independent queries or independent data-source counts.
The training_curves.png bands are descriptive quartiles across dependent fitted heads, not confidence intervals or held-source generalization evidence.

The four outputs are signed-risk score bases, not identified calibrated moments. Query means are supervised on known labels only, while all causal rows remain in inference.
Unknown outcomes stay unknown; incomplete fixed-roster risk summaries do not become zero risk.
Three thousand locality-bootstrap draws follow averaging of dependent producer/fit/seed views. No IID-window or independent-confirmation claim.
These are retained query cohorts, not a completeness guarantee for all visible agents. No pairwise collision term was learned.
Image-local detector silver, obs8/pred12 raw-frame stride12. No metric, seconds, human-gold, physical-safety, true3D or foundation claim. No Stage5C or SMC.
