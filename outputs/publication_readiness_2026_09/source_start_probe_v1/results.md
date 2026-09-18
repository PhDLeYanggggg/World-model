# Source-Supported Start Information

45 fresh classifiers, 54 prediction cells; fixed exposed-fit sites only. Not a forecasting or deployment result.

| Model / source | Held site | Brier lift vs main prior | Brier lift vs main-only | AUROC | Positive seeds |
| --- | --- | ---: | ---: | ---: | ---: |
| logistic_main_only | ETH | -0.009816 | 0.000000 | 0.598613 | 0/3 |
| logistic_main_only | Hotel | -0.167381 | 0.000000 | 0.411628 | 0/3 |
| logistic_source_only | ETH | -0.071974 | -0.062158 | 0.408320 | 0/3 |
| logistic_source_only | Hotel | 0.068868 | 0.236249 | 0.550563 | 3/3 |
| logistic_mixed | ETH | -0.015908 | -0.006092 | 0.556240 | 0/3 |
| logistic_mixed | Hotel | 0.003845 | 0.171226 | 0.496574 | 3/3 |
| extra_trees_main_only | ETH | 0.008525 | 0.000000 | 0.611710 | 3/3 |
| extra_trees_main_only | Hotel | -0.103452 | 0.000000 | 0.376644 | 0/3 |
| extra_trees_source_only | ETH | -0.110321 | -0.118846 | 0.516179 | 0/3 |
| extra_trees_source_only | Hotel | 0.066763 | 0.170215 | 0.560799 | 3/3 |
| extra_trees_mixed | ETH | -0.054370 | -0.062895 | 0.486389 | 0/3 |
| extra_trees_mixed | Hotel | 0.063793 | 0.167245 | 0.517304 | 3/3 |
| mlp_main_only | ETH | -0.000385 | 0.000000 | 0.737288 | 2/3 |
| mlp_main_only | Hotel | -0.183523 | 0.000000 | 0.451646 | 0/3 |
| mlp_source_only | ETH | -0.119195 | -0.118810 | 0.345403 | 0/3 |
| mlp_source_only | Hotel | 0.058992 | 0.242515 | 0.553463 | 3/3 |
| mlp_mixed | ETH | -0.112763 | -0.112378 | 0.561633 | 0/3 |
| mlp_mixed | Hotel | 0.024074 | 0.207597 | 0.579670 | 3/3 |

Zara has no exactly-stationary rows: not_run for this information probe, not another passing domain.
No forecast is switched. Held-agent intervals are descriptive, not new-scene confirmation or calibration.
Source-only model inputs, normalization and fit use no main data; the reported comparator prior uses the opposite main training site.
Offline/silver, dataset-local/raw annotation steps only. No metric, seconds, foundation, Stage5C or SMC.
