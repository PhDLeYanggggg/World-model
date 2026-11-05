# Complete Fitting Diagnostics

Empirical whole-query draw weights. Scores are not calibrated bounds.
Logistic-score MSE is a cross-objective diagnostic, not a risk calibration metric.

| Context | Objective | Features | MSE | Signed-cost objective | All/easy sign regret | False-safe all/easy |
|---|---|---|---:|---:|---|---|
| eqmotion_fold1_seed17_critic0 | continued_mse | unmasked | 0.01193634 | 0.04967311 | [0.004702069115601703, 0.0016522157767175718] | [127, 76] |
| eqmotion_fold1_seed17_critic0 | continued_mse | scale_ablated | 0.01776455 | 0.05226510 | [0.005712012032875739, 0.001908526551611173] | [117, 72] |
| eqmotion_fold1_seed17_critic0 | signed_cost_logistic | unmasked | 20.67499845 | 0.00563478 | [0.0004031908283131536, 0.0003133441502116798] | [81, 23] |
| eqmotion_fold1_seed17_critic0 | signed_cost_logistic | scale_ablated | 22.74636519 | 0.00517445 | [0.0003623955193915798, 0.00031264161800675623] | [75, 24] |
| eqmotion_fold0_seed17_critic1 | continued_mse | unmasked | 0.01163633 | 0.04565644 | [0.003915805499679094, 0.0004650252474167702] | [225, 73] |
| eqmotion_fold0_seed17_critic1 | continued_mse | scale_ablated | 0.01123570 | 0.04634453 | [0.003974927892470117, 0.0005543328975252512] | [268, 81] |
| eqmotion_fold0_seed17_critic1 | signed_cost_logistic | unmasked | 9.06812702 | 0.00608023 | [0.0008932923522066469, 7.870592637835035e-05] | [160, 15] |
| eqmotion_fold0_seed17_critic1 | signed_cost_logistic | scale_ablated | 9.60233124 | 0.00608861 | [0.0009541586040215807, 8.08922942755168e-05] | [154, 20] |
| eqmotion_fold0_seed17_critic2 | continued_mse | unmasked | 0.68002385 | 0.61500546 | [0.01949702105119334, 0.006812996655288036] | [2804, 952] |
| eqmotion_fold0_seed17_critic2 | continued_mse | scale_ablated | 0.68881004 | 0.61061286 | [0.01944078957363299, 0.006540473374103489] | [2530, 1024] |
| eqmotion_fold0_seed17_critic2 | signed_cost_logistic | unmasked | 24.32517650 | 0.08344438 | [0.010537859025613371, 0.0010149240217509278] | [838, 43] |
| eqmotion_fold0_seed17_critic2 | signed_cost_logistic | scale_ablated | 25.33364513 | 0.10527362 | [0.011160666789868916, 0.0010218261609309584] | [679, 39] |

## Prospective Context Screens

| Objective | Context | Completion lower pp | Expansion |
|---|---|---:|---|
| continued_mse | eqmotion_fold1_seed17_critic0 | 0.9998 | True |
| continued_mse | eqmotion_fold0_seed17_critic1 | 0.3755 | True |
| continued_mse | eqmotion_fold0_seed17_critic2 | -0.6966 | False |
| signed_cost_logistic | eqmotion_fold1_seed17_critic0 | 0.4486 | True |
| signed_cost_logistic | eqmotion_fold0_seed17_critic1 | -0.0416 | False |
| signed_cost_logistic | eqmotion_fold0_seed17_critic2 | -0.0051 | False |
