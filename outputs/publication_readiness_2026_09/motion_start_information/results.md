# Observed Motion and Start Probability

48 fresh classifier fits; no new trajectory or neural model. All values below are seed means.
Brier differences are absolute score differences, not percentages or ADE/FDE gains.

| Held site | Model | Inputs | Train Brier lift | Held Brier lift | Held AUC | Lift vs quality | Positive seeds |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| ETH | logistic | neighbors | 0.05512 | -0.20060 | 0.2766 | 0.03422 | 0/3 |
| ETH | extra_trees | neighbors | 0.14110 | -0.03195 | 0.6692 | -0.00010 | 0/3 |
| ETH | logistic | quality | 0.06066 | -0.23481 | 0.2411 | 0.00000 | 0/3 |
| ETH | extra_trees | quality | 0.14144 | -0.03185 | 0.6857 | 0.00000 | 0/3 |
| ETH | logistic | magnitude | 0.10528 | -0.14654 | 0.3952 | 0.08828 | 0/3 |
| ETH | extra_trees | magnitude | 0.15332 | 0.08030 | 0.8105 | 0.11215 | 3/3 |
| ETH | logistic | directed | 0.11778 | -0.13072 | 0.4284 | 0.10409 | 0/3 |
| ETH | extra_trees | directed | 0.15877 | 0.07870 | 0.8215 | 0.11055 | 3/3 |
| Hotel | logistic | neighbors | 0.12238 | -0.10187 | 0.5041 | -0.00322 | 0/3 |
| Hotel | extra_trees | neighbors | 0.11302 | -0.00820 | 0.5003 | 0.00050 | 0/3 |
| Hotel | logistic | quality | 0.12297 | -0.09865 | 0.5080 | 0.00000 | 0/3 |
| Hotel | extra_trees | quality | 0.11208 | -0.00870 | 0.4878 | 0.00000 | 0/3 |
| Hotel | logistic | magnitude | 0.14923 | -0.10371 | 0.4708 | -0.00506 | 0/3 |
| Hotel | extra_trees | magnitude | 0.12015 | -0.00484 | 0.5004 | 0.00386 | 0/3 |
| Hotel | logistic | directed | 0.18378 | -0.10495 | 0.4717 | -0.00630 | 0/3 |
| Hotel | extra_trees | directed | 0.12263 | -0.00101 | 0.5118 | 0.00769 | 1/3 |

## Group-Balanced Diagnostic

Intervals resample held agents within the same previously exposed site. Five ETH and 26 Hotel agents
do not constitute new-scene confirmation. Seed losses are averaged; probabilities are not ensembled.

| Held | Model | Inputs | Agent lift vs prior [95% interval] | Agent lift vs quality [95% interval] | Positive agents vs prior | Run lift vs prior |
| --- | --- | --- | --- | --- | ---: | ---: |
| ETH | logistic | neighbors | -0.20567 [-0.25066, -0.16068] | 0.04699 [-0.01141, 0.13364] | 0/5 | -0.20459 |
| ETH | extra_trees | neighbors | -0.00882 [-0.10063, 0.06790] | 0.00039 [-0.00177, 0.00222] | 3/5 | -0.04570 |
| ETH | logistic | quality | -0.25266 [-0.31743, -0.18791] | 0.00000 [0.00000, 0.00000] | 0/5 | -0.26855 |
| ETH | extra_trees | quality | -0.00921 [-0.10220, 0.06793] | 0.00000 [0.00000, 0.00000] | 3/5 | -0.04479 |
| ETH | logistic | magnitude | -0.17999 [-0.28414, -0.08430] | 0.07267 [-0.06608, 0.21407] | 0/5 | -0.11422 |
| ETH | extra_trees | magnitude | 0.08960 [0.02276, 0.15643] | 0.09881 [-0.02856, 0.24361] | 5/5 | 0.12492 |
| ETH | logistic | directed | -0.11499 [-0.18739, -0.06762] | 0.13766 [0.04987, 0.22546] | 0/5 | -0.09692 |
| ETH | extra_trees | directed | 0.08642 [0.01832, 0.15451] | 0.09563 [-0.03481, 0.24018] | 5/5 | 0.12132 |
| Hotel | logistic | neighbors | -0.06482 [-0.13941, 0.01437] | -0.00785 [-0.03246, 0.00642] | 10/26 | -0.14137 |
| Hotel | extra_trees | neighbors | 0.01359 [-0.04656, 0.07966] | -0.00027 [-0.00446, 0.00420] | 13/26 | 0.00325 |
| Hotel | logistic | quality | -0.05697 [-0.13189, 0.02063] | 0.00000 [0.00000, 0.00000] | 11/26 | -0.13528 |
| Hotel | extra_trees | quality | 0.01385 [-0.04502, 0.07939] | 0.00000 [0.00000, 0.00000] | 14/26 | 0.00303 |
| Hotel | logistic | magnitude | -0.06204 [-0.15659, 0.04914] | -0.00507 [-0.07473, 0.05636] | 8/26 | -0.13286 |
| Hotel | extra_trees | magnitude | 0.01635 [-0.04313, 0.08323] | 0.00250 [-0.00272, 0.00829] | 13/26 | 0.00503 |
| Hotel | logistic | directed | -0.05293 [-0.15626, 0.06792] | 0.00404 [-0.09289, 0.10457] | 10/26 | -0.12905 |
| Hotel | extra_trees | directed | 0.01801 [-0.03993, 0.08207] | 0.00416 [-0.00296, 0.01160] | 14/26 | 0.00622 |

Zara: not_run, no exactly-static histories. No model/threshold chosen; full forecasting cohort unchanged.
All outputs remain dataset-local/native-step diagnostics. No new deployment, Stage5C or SMC.
