# Fitting Objectives And Decision Errors

Complete empirical query weights, not one minibatch. MSE columns applied
to logistic scores are only cross-objective diagnostics, not calibrated
risk errors. Count/regret units are repeated fitting channels.

| Context | Model | MSE | Signed-cost mixed objective | All/easy sign regret | False-safe all/easy |
|---|---|---:|---:|---|---|
| eqmotion_fold1_seed17_critic0 | warm | 0.02024887 | 0.05299937 | [0.005885331320739159, 0.0018230697801884432] | [92, 51] |
| eqmotion_fold1_seed17_critic0 | continued_mse | 0.01193634 | 0.04967311 | [0.004702069115601703, 0.0016522157767175718] | [127, 76] |
| eqmotion_fold1_seed17_critic0 | signed_cost_logistic | 20.67499845 | 0.00563478 | [0.0004031908283131536, 0.0003133441502116798] | [81, 23] |
| eqmotion_fold0_seed17_critic1 | warm | 0.01896097 | 0.05097955 | [0.0040500359741598765, 0.0006295278165727729] | [290, 75] |
| eqmotion_fold0_seed17_critic1 | continued_mse | 0.01163633 | 0.04565644 | [0.003915805499679094, 0.0004650252474167702] | [225, 73] |
| eqmotion_fold0_seed17_critic1 | signed_cost_logistic | 9.06812702 | 0.00608023 | [0.0008932923522066469, 7.870592637835035e-05] | [160, 15] |
| eqmotion_fold0_seed17_critic2 | warm | 0.89371054 | 0.75667169 | [0.020061782425412445, 0.008376344530067172] | [2219, 876] |
| eqmotion_fold0_seed17_critic2 | continued_mse | 0.68002385 | 0.61500546 | [0.01949702105119334, 0.006812996655288036] | [2804, 952] |
| eqmotion_fold0_seed17_critic2 | signed_cost_logistic | 24.32517650 | 0.08344438 | [0.010537859025613371, 0.0010149240217509278] | [838, 43] |

## Prospective Context Results

| Context | Paired completion lower pp | Expansion condition |
|---|---:|---|
| eqmotion_fold1_seed17_critic0 | 5.0852 | True |
| eqmotion_fold0_seed17_critic1 | -11.9151 | False |
| eqmotion_fold0_seed17_critic2 | -0.7047 | False |

No favorable-source selection or changed threshold follows this readout.
The ideal fully observed conditional optimum derivation is an objective
motivation, not a finite-network generalization or safety theorem.
