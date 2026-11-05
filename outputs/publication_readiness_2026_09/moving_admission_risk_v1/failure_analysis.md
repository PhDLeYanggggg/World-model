# Priority Learning: Fitting And Held Risk

The fixed expansion condition is False.

## Fitting Policy, Resubstitution Only

| Policy | Known floor gain % | Hard gain % | Switch % | Full risk/support | All/easy violations | Undefined selected easy |
|---|---:|---:|---:|---:|---|---:|
| mean_veto | 6.992155 | 7.782569 | 22.115276 | 2/24 | 15/15 | 6 |
| readout_veto | 7.722350 | 8.555668 | 24.195528 | 2/24 | 17/18 | 4 |
| kinematic_veto | 7.411568 | 7.345708 | 21.788208 | 2/24 | 13/15 | 6 |
| continued_mse_veto | 8.291558 | 8.267473 | 24.274425 | 2/24 | 15/17 | 5 |
| false_safe_priority_veto | 7.792348 | 7.291569 | 20.101654 | 4/24 | 12/14 | 6 |

## All Fixed Contexts

| Context | Priority-control completion lower pp | Control all/easy violations | Priority all/easy violations | Priority undefined easy | Expansion |
|---|---:|---|---|---:|---|
| eqmotion_fold1_seed17_critic0 | 0.205596 | 9/9 | 9/9 | 0 | True |
| eqmotion_fold0_seed17_critic1 | -2.673935 | 3/4 | 3/4 | 1 | False |
| eqmotion_fold0_seed17_critic2 | -0.255420 | 2/0 | 2/0 | 1 | False |

## Conditional Risk

Pooled known-label native-coordinate sums over repeated context views, not
independent samples, official source-macro ratios or calibration certificates.
Unknown labels retain bounds in the complete evaluation.

| Scope | Arm | Set | Known / unknown | Predicted all/easy excess | Actual all/easy excess |
|---|---|---|---:|---|---|
| fit | kinematic | proposals | 48061/1274 | 354786.749646, 158211.446523 | 235500.783506, 45001.300410 |
| fit | kinematic | admitted | 11696/59 | -57416.535720, -55669.678365 | -3770.592630, 1405.253435 |
| fit | continued_mse | proposals | 48061/1274 | 309980.324730, 119004.291397 | 235500.783506, 45001.300410 |
| fit | continued_mse | admitted | 14044/76 | -67868.667224, -62705.292877 | -2952.539552, 1543.246141 |
| fit | false_safe_priority | proposals | 48061/1274 | 413733.005062, 168844.940873 | 235500.783506, 45001.300410 |
| fit | false_safe_priority | admitted | 8639/41 | -35148.531639, -39331.551136 | -9738.706758, 320.581198 |
| held_SOURCE | kinematic | proposals | 78665/1444 | 867579.299618, 772472.772330 | 63859.061813, 37718.028649 |
| held_SOURCE | kinematic | admitted | 29077/530 | -228030.031529, -250999.258863 | 12696.431595, 10190.137944 |
| held_SOURCE | continued_mse | proposals | 78665/1444 | 1057753.623084, 982277.466564 | 63859.061813, 37718.028649 |
| held_SOURCE | continued_mse | admitted | 24160/442 | -210008.831652, -226812.668399 | 14561.592013, 9777.866274 |
| held_SOURCE | false_safe_priority | proposals | 78665/1444 | 1077132.044114, 929984.157778 | 63859.061813, 37718.028649 |
| held_SOURCE | false_safe_priority | admitted | 21218/378 | -195003.217941, -229689.514767 | 13032.922077, 9508.291487 |

## Limits

Fixed warm-error emphasis may miss new errors introduced by training and
can discard useful actions. Weight 10 is fixed, not tuned to this readout.
The warm error mask is defined by fitting-data resubstitution, not by an
independent error-discovery split or out-of-fold risk predictions.
Lower fitting loss, fewer interventions or a favorable source cannot
replace complete original risk/support. Weighted outputs are not unbiased
expected harm. Old online-admission, gradient-parity, quantile and subset
failures remain negative; this single contrast does not erase them.

Model training cannot create missing full easy-reference support or restore
independence to exposed SOURCE. Original primary NA and all old failures
remain. No new H, deployment change, Stage5C, SMC or submission-ready claim.
