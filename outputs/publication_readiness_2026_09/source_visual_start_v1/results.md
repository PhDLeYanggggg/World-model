# Matched Visual Start-Information Results

All means average seed losses, not prediction ensembles. Positive lift means lower Brier, not ADE/FDE.
The primary contrast is RGB versus the same-schedule coverage control. All sites are exposed fit roles.

| Arm | Schedule | Held site | Brier | Lift vs mask | Lift vs own train prior | Lift vs source prior | AUROC | ECE |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mask_only | main_only | ETH | 0.231072 | 0.000000 | 0.041932 | 0.044998 | 0.809707 | 0.240489 |
| mask_only | main_only | Hotel | 0.516946 | 0.000000 | -0.193872 | -0.269011 | 0.428407 | 0.518008 |
| mask_only | source_only | ETH | 0.443754 | 0.000000 | -0.167684 | -0.167684 | 0.421931 | 0.442130 |
| mask_only | source_only | Hotel | 0.270223 | 0.000000 | -0.022288 | -0.022288 | 0.557648 | 0.146130 |
| mask_only | mixed | ETH | 0.316904 | 0.000000 | -0.042374 | -0.040834 | 0.708269 | 0.350297 |
| mask_only | mixed | Hotel | 0.302667 | 0.000000 | -0.036721 | -0.054731 | 0.559115 | 0.184670 |
| past_rgb | main_only | ETH | 0.214901 | 0.016172 | 0.058104 | 0.061170 | 0.821777 | 0.218145 |
| past_rgb | main_only | Hotel | 0.519182 | -0.002236 | -0.196108 | -0.271246 | 0.430108 | 0.519609 |
| past_rgb | source_only | ETH | 0.534536 | -0.090782 | -0.258466 | -0.258466 | 0.425783 | 0.555092 |
| past_rgb | source_only | Hotel | 0.267971 | 0.002252 | -0.020036 | -0.020036 | 0.570426 | 0.129667 |
| past_rgb | mixed | ETH | 0.277526 | 0.039378 | -0.002996 | -0.001456 | 0.565229 | 0.275868 |
| past_rgb | mixed | Hotel | 0.266622 | 0.036045 | -0.000676 | -0.018686 | 0.562991 | 0.123592 |

## RGB Incremental Uncertainty

The intervals below are conditional agent-balanced contrasts; row means above have a different estimand.
Five ETH and 26 Hotel IDs, overlapping windows and two exposed sites do not establish independent scene generalization.

| Schedule | Held site | Agent-balanced RGB lift [95% interval] | Row-mean bias-shift term | Row-mean varying-prediction term | Positive seeds |
| --- | --- | --- | ---: | ---: | ---: |
| main_only | ETH | 0.030703 [-0.007703, 0.069109] | 0.006533 | 0.009639 | 2/3 |
| main_only | Hotel | -0.003200 [-0.008234, 0.000791] | -0.002622 | 0.000387 | 0/3 |
| source_only | ETH | -0.103804 [-0.135363, -0.072244] | -0.098114 | 0.007333 | 0/3 |
| source_only | Hotel | -0.000289 [-0.009488, 0.008462] | -0.000148 | 0.002401 | 2/3 |
| mixed | ETH | -0.019117 [-0.133658, 0.083091] | 0.079563 | -0.040185 | 3/3 |
| mixed | Hotel | -0.038780 [-0.149999, 0.062546] | -0.000793 | 0.036838 | 3/3 |

Brier decomposition uses held labels only to explain fixed predictions, never to recalibrate or select.
No forecast policy, physical-time, metric, true-3D, foundation, Stage5C or SMC claim.

## Descriptive Influence Check

Supplementary, not a registered selection rule: omit one local agent ID at a time.
The range is sensitivity, not a confidence interval; shared scene dependence remains.

| Schedule | Held site | Minimum omitted-agent lift | Maximum omitted-agent lift | Positive omissions |
| --- | --- | ---: | ---: | ---: |
| main_only | ETH | 0.016564 | 0.045299 | 5/5 |
| main_only | Hotel | -0.003892 | -0.001585 | 0/26 |
| source_only | ETH | -0.116370 | -0.092480 | 0/5 |
| source_only | Hotel | -0.002115 | 0.001835 | 10/26 |
| mixed | ETH | -0.054085 | 0.025354 | 2/5 |
| mixed | Hotel | -0.050946 | -0.009975 | 0/26 |
