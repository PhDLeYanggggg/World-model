# Fixed-Matrix Probability Contrasts

Positive Brier lift means lower squared probability error. It is not trajectory improvement.
Intervals give equal weight to held agents; main point estimates give equal weight to rows.
These are different descriptive estimands. Neither is a new-site confirmation interval.

| Arm | Held site | Brier lift vs main prior | Agent-balanced lift vs prior [95% interval] | Lift vs main-only | Agent-balanced lift vs main [95% interval] |
| --- | --- | ---: | --- | ---: | --- |
| logistic_main_only | ETH | -0.009816 | -0.035460 [-0.170599, 0.095748] | 0.000000 | 0.000000 [0.000000, 0.000000] |
| logistic_main_only | Hotel | -0.167381 | -0.072029 [-0.139883, -0.003396] | 0.000000 | 0.000000 [0.000000, 0.000000] |
| logistic_source_only | ETH | -0.071974 | -0.118842 [-0.200261, -0.047696] | -0.062158 | -0.083382 [-0.178408, 0.012169] |
| logistic_source_only | Hotel | 0.068868 | 0.027322 [-0.075412, 0.136539] | 0.236249 | 0.099351 [-0.040653, 0.239797] |
| logistic_mixed | ETH | -0.015908 | -0.042954 [-0.153856, 0.064761] | -0.006092 | -0.007494 [-0.038431, 0.023442] |
| logistic_mixed | Hotel | 0.003845 | 0.039623 [-0.002767, 0.087113] | 0.171226 | 0.111651 [0.049507, 0.177292] |
| extra_trees_main_only | ETH | 0.008525 | 0.038452 [-0.022290, 0.099194] | 0.000000 | 0.000000 [0.000000, 0.000000] |
| extra_trees_main_only | Hotel | -0.103452 | -0.048995 [-0.098391, -0.006947] | 0.000000 | 0.000000 [0.000000, 0.000000] |
| extra_trees_source_only | ETH | -0.110321 | -0.199567 [-0.368248, -0.057190] | -0.118846 | -0.238019 [-0.439875, -0.069070] |
| extra_trees_source_only | Hotel | 0.066763 | 0.004839 [-0.105801, 0.111933] | 0.170215 | 0.053835 [-0.086160, 0.191261] |
| extra_trees_mixed | ETH | -0.054370 | -0.094812 [-0.187265, -0.002360] | -0.062895 | -0.133264 [-0.247344, -0.028821] |
| extra_trees_mixed | Hotel | 0.063793 | 0.033718 [-0.042240, 0.114107] | 0.167245 | 0.082714 [-0.021759, 0.188460] |
| mlp_main_only | ETH | -0.000385 | 0.092694 [-0.031357, 0.216746] | 0.000000 | 0.000000 [0.000000, 0.000000] |
| mlp_main_only | Hotel | -0.183523 | -0.118156 [-0.196816, -0.028019] | 0.000000 | 0.000000 [0.000000, 0.000000] |
| mlp_source_only | ETH | -0.119195 | -0.188785 [-0.312408, -0.090666] | -0.118810 | -0.281480 [-0.532953, -0.062492] |
| mlp_source_only | Hotel | 0.058992 | 0.012541 [-0.100510, 0.126877] | 0.242515 | 0.130697 [-0.028560, 0.287755] |
| mlp_mixed | ETH | -0.112763 | -0.081094 [-0.218717, 0.048930] | -0.112378 | -0.173788 [-0.294423, -0.053154] |
| mlp_mixed | Hotel | 0.024074 | 0.056029 [-0.003024, 0.121478] | 0.207597 | 0.174185 [0.079626, 0.277145] |

Source constant-prior scores in fit_metrics.csv are post hoc analytic diagnostics, not another selected classifier.
No independence claim for overlapping windows, contemporaneous agents or repeated fits.
No pooled best arm, intervention, threshold search or independent confirmation.
