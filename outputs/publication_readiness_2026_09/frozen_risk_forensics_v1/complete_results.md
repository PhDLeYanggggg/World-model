# All Frozen Risk Comparisons

Fresh descriptive reduction of cached, hash-verified frozen decisions. No fitting,
policy selection, independent calibration or new primary metric. Each row reuses
the same 970 queries/37,775 past agents, including 28,324 ADE-labeled agents.
These are repeated comparisons, not independent sample sizes.

E = observed lower bound proves the realized budget is exceeded; W = every
selected cost is known and the realized budget is met; ? = indeterminate.
Missing selected costs are never imputed as zero. W is an outcome-defined
diagnostic, not a causal deployment filter. All predicted budgets pass.

| Family | Fixed combination | Control | E / W / ? queries | Missing selected costs | Mean predicted harm | Mean observed lower bound | Easy harm from W (%) | Easy degradation (%) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| transformer | seed17_ridge_conservative | risk-only | 133 / 344 / 493 | 1144 | 0.00023999641 | 0.015174591 | 1.8670676 | 2474.4596 |
| transformer | seed17_ridge_conservative | unary | 133 / 344 / 493 | 1144 | 0.00023999641 | 0.015174591 | 1.8670676 | 2474.4596 |
| transformer | seed17_ridge_conservative | joint | 133 / 344 / 493 | 1144 | 0.00023999641 | 0.015174591 | 1.8670676 | 2474.4596 |
| transformer | seed17_ridge_moderate | risk-only | 102 / 302 / 566 | 1394 | 0.00056271846 | 0.015961104 | 1.9472099 | 2620.5033 |
| transformer | seed17_ridge_moderate | unary | 102 / 302 / 566 | 1394 | 0.00056271846 | 0.015961104 | 1.9472099 | 2620.5033 |
| transformer | seed17_ridge_moderate | joint | 102 / 302 / 566 | 1394 | 0.00056271846 | 0.015961104 | 1.9472099 | 2620.5033 |
| transformer | seed17_neural_cost_conservative | risk-only | 16 / 543 / 411 | 616 | 0.00078203631 | 0.00078226563 | 17.065491 | 107.26203 |
| transformer | seed17_neural_cost_conservative | unary | 16 / 543 / 411 | 616 | 0.00078203631 | 0.00078226563 | 17.065491 | 107.26203 |
| transformer | seed17_neural_cost_conservative | joint | 16 / 543 / 411 | 616 | 0.00078203631 | 0.00078226563 | 17.065491 | 107.26203 |
| transformer | seed17_neural_cost_moderate | risk-only | 1 / 383 / 586 | 1121 | 0.0027745637 | 0.0014950886 | 18.757792 | 178.8628 |
| transformer | seed17_neural_cost_moderate | unary | 1 / 383 / 586 | 1121 | 0.0027745637 | 0.0014950886 | 18.757792 | 178.8628 |
| transformer | seed17_neural_cost_moderate | joint | 1 / 383 / 586 | 1121 | 0.0027745637 | 0.0014950886 | 18.757792 | 178.8628 |
| transformer | seed29_ridge_conservative | risk-only | 33 / 301 / 636 | 1360 | 0.00025067675 | 0.0037730066 | 6.4094905 | 548.4971 |
| transformer | seed29_ridge_conservative | unary | 33 / 301 / 636 | 1361 | 0.00025042134 | 0.0037730066 | 6.4094905 | 548.4971 |
| transformer | seed29_ridge_conservative | joint | 33 / 301 / 636 | 1360 | 0.00025067675 | 0.0037730066 | 6.4094905 | 548.4971 |
| transformer | seed29_ridge_moderate | risk-only | 23 / 162 / 785 | 2024 | 0.00060251229 | 0.0051891002 | 5.3454869 | 669.38119 |
| transformer | seed29_ridge_moderate | unary | 23 / 162 / 785 | 2024 | 0.00060251229 | 0.0051891002 | 5.3454869 | 669.38119 |
| transformer | seed29_ridge_moderate | joint | 23 / 162 / 785 | 2024 | 0.00060251229 | 0.0051891002 | 5.3454869 | 669.38119 |
| transformer | seed29_neural_cost_conservative | risk-only | 0 / 692 / 278 | 328 | 0.00043293381 | 0.0003374879 | 51.750251 | 31.30719 |
| transformer | seed29_neural_cost_conservative | unary | 0 / 692 / 278 | 328 | 0.00043293381 | 0.0003374879 | 51.750251 | 31.30719 |
| transformer | seed29_neural_cost_conservative | joint | 0 / 692 / 278 | 328 | 0.00043293381 | 0.0003374879 | 51.750251 | 31.30719 |
| transformer | seed29_neural_cost_moderate | risk-only | 0 / 567 / 403 | 578 | 0.0013604481 | 0.00071781578 | 44.805964 | 69.888057 |
| transformer | seed29_neural_cost_moderate | unary | 0 / 567 / 403 | 578 | 0.0013604481 | 0.00071781578 | 44.805964 | 69.888057 |
| transformer | seed29_neural_cost_moderate | joint | 0 / 567 / 403 | 578 | 0.0013604481 | 0.00071781578 | 44.805964 | 69.888057 |
| transformer | seed43_ridge_conservative | risk-only | 107 / 732 / 131 | 197 | 2.8098581e-05 | 0.012435204 | 2.7488621 | 1874.225 |
| transformer | seed43_ridge_conservative | unary | 107 / 732 / 131 | 197 | 2.8098581e-05 | 0.012435204 | 2.7488621 | 1874.225 |
| transformer | seed43_ridge_conservative | joint | 107 / 732 / 131 | 197 | 2.8098581e-05 | 0.012435204 | 2.7488621 | 1874.225 |
| transformer | seed43_ridge_moderate | risk-only | 104 / 697 / 169 | 268 | 7.2933998e-05 | 0.012884706 | 3.7605445 | 1953.6997 |
| transformer | seed43_ridge_moderate | unary | 104 / 697 / 169 | 268 | 7.2933998e-05 | 0.012884706 | 3.7605445 | 1953.6997 |
| transformer | seed43_ridge_moderate | joint | 104 / 697 / 169 | 268 | 7.2933998e-05 | 0.012884706 | 3.7605445 | 1953.6997 |
| transformer | seed43_neural_cost_conservative | risk-only | 13 / 799 / 158 | 183 | 0.0005080161 | 0.0010319148 | 57.877568 | 146.95806 |
| transformer | seed43_neural_cost_conservative | unary | 13 / 799 / 158 | 183 | 0.0005080161 | 0.0010319148 | 57.877568 | 146.95806 |
| transformer | seed43_neural_cost_conservative | joint | 13 / 799 / 158 | 183 | 0.0005080161 | 0.0010319148 | 57.877568 | 146.95806 |
| transformer | seed43_neural_cost_moderate | risk-only | 6 / 673 / 291 | 407 | 0.0014453398 | 0.0021422639 | 50.248624 | 339.1927 |
| transformer | seed43_neural_cost_moderate | unary | 6 / 673 / 291 | 407 | 0.0014453398 | 0.0021422639 | 50.248624 | 339.1927 |
| transformer | seed43_neural_cost_moderate | joint | 6 / 673 / 291 | 407 | 0.0014453398 | 0.0021422639 | 50.248624 | 339.1927 |
| eqmotion | seed17_ridge_conservative | risk-only | 234 / 301 / 435 | 1050 | 6.4159704e-05 | 0.081094413 | 0.28929948 | 16349.812 |
| eqmotion | seed17_ridge_conservative | unary | 234 / 301 / 435 | 1050 | 6.4159704e-05 | 0.081094413 | 0.28929948 | 16349.812 |
| eqmotion | seed17_ridge_conservative | joint | 234 / 301 / 435 | 1050 | 6.4159704e-05 | 0.081094413 | 0.28929948 | 16349.812 |
| eqmotion | seed17_ridge_moderate | risk-only | 212 / 254 / 504 | 1275 | 0.0001759572 | 0.082138027 | 0.4036099 | 16400.894 |
| eqmotion | seed17_ridge_moderate | unary | 212 / 254 / 504 | 1275 | 0.0001759572 | 0.082138027 | 0.4036099 | 16400.894 |
| eqmotion | seed17_ridge_moderate | joint | 212 / 254 / 504 | 1275 | 0.0001759572 | 0.082138027 | 0.4036099 | 16400.894 |
| eqmotion | seed17_neural_cost_conservative | risk-only | 7 / 922 / 41 | 45 | 0.00012890271 | 0.00075550021 | 6.6998869 | 124.36572 |
| eqmotion | seed17_neural_cost_conservative | unary | 7 / 922 / 41 | 45 | 0.00012890271 | 0.00075550021 | 6.6998869 | 124.36572 |
| eqmotion | seed17_neural_cost_conservative | joint | 7 / 922 / 41 | 45 | 0.00012890271 | 0.00075550021 | 6.6998869 | 124.36572 |
| eqmotion | seed17_neural_cost_moderate | risk-only | 12 / 826 / 132 | 161 | 0.00067828702 | 0.0013955266 | 12.271778 | 183.99383 |
| eqmotion | seed17_neural_cost_moderate | unary | 12 / 826 / 132 | 161 | 0.00067828702 | 0.0013955266 | 12.271778 | 183.99383 |
| eqmotion | seed17_neural_cost_moderate | joint | 12 / 826 / 132 | 161 | 0.00067828702 | 0.0013955266 | 12.271778 | 183.99383 |
| eqmotion | seed29_ridge_conservative | risk-only | 528 / 71 / 371 | 1890 | 0.00018084755 | 0.099820198 | 0.050404942 | 21597.515 |
| eqmotion | seed29_ridge_conservative | unary | 529 / 71 / 370 | 1886 | 0.00018058335 | 0.099824834 | 0.050403625 | 21598.08 |
| eqmotion | seed29_ridge_conservative | joint | 529 / 71 / 370 | 1888 | 0.00018108562 | 0.099817219 | 0.050403625 | 21598.08 |
| eqmotion | seed29_ridge_moderate | risk-only | 348 / 43 / 579 | 3530 | 0.0011393291 | 0.10364404 | 0.038954641 | 21821.611 |
| eqmotion | seed29_ridge_moderate | unary | 348 / 43 / 579 | 3529 | 0.001136242 | 0.10363475 | 0.038953325 | 21822.349 |
| eqmotion | seed29_ridge_moderate | joint | 348 / 43 / 579 | 3531 | 0.0011372176 | 0.10363568 | 0.038954641 | 21821.611 |
| eqmotion | seed29_neural_cost_conservative | risk-only | 32 / 669 / 269 | 340 | 0.00050306065 | 0.0024619687 | 7.3614022 | 517.11053 |
| eqmotion | seed29_neural_cost_conservative | unary | 32 / 669 / 269 | 340 | 0.00050306065 | 0.0024619687 | 7.3614022 | 517.11053 |
| eqmotion | seed29_neural_cost_conservative | joint | 32 / 669 / 269 | 340 | 0.00050306065 | 0.0024619687 | 7.3614022 | 517.11053 |
| eqmotion | seed29_neural_cost_moderate | risk-only | 29 / 558 / 383 | 568 | 0.0015103308 | 0.0054162585 | 5.5919339 | 1140.0653 |
| eqmotion | seed29_neural_cost_moderate | unary | 29 / 558 / 383 | 568 | 0.0015103308 | 0.0054162585 | 5.5919339 | 1140.0653 |
| eqmotion | seed29_neural_cost_moderate | joint | 29 / 558 / 383 | 568 | 0.0015103308 | 0.0054162585 | 5.5919339 | 1140.0653 |
| eqmotion | seed43_ridge_conservative | risk-only | 110 / 432 / 428 | 786 | 3.9763248e-05 | 0.022712889 | 0.55336153 | 3639.0895 |
| eqmotion | seed43_ridge_conservative | unary | 110 / 432 / 428 | 786 | 3.9763248e-05 | 0.022712889 | 0.55336153 | 3639.0895 |
| eqmotion | seed43_ridge_conservative | joint | 110 / 432 / 428 | 786 | 3.9763248e-05 | 0.022712889 | 0.55336153 | 3639.0895 |
| eqmotion | seed43_ridge_moderate | risk-only | 105 / 413 / 452 | 850 | 0.00013499142 | 0.023471267 | 0.57532384 | 3824.5685 |
| eqmotion | seed43_ridge_moderate | unary | 105 / 413 / 452 | 850 | 0.00013499142 | 0.023471267 | 0.57532384 | 3824.5685 |
| eqmotion | seed43_ridge_moderate | joint | 105 / 413 / 452 | 850 | 0.00013499142 | 0.023471267 | 0.57532384 | 3824.5685 |
| eqmotion | seed43_neural_cost_conservative | risk-only | 27 / 734 / 209 | 285 | 0.0005577412 | 0.002576324 | 16.323352 | 452.69186 |
| eqmotion | seed43_neural_cost_conservative | unary | 27 / 734 / 209 | 285 | 0.0005577412 | 0.002576324 | 16.323352 | 452.69186 |
| eqmotion | seed43_neural_cost_conservative | joint | 27 / 734 / 209 | 285 | 0.0005577412 | 0.002576324 | 16.323352 | 452.69186 |
| eqmotion | seed43_neural_cost_moderate | risk-only | 16 / 645 / 309 | 473 | 0.0013165784 | 0.0039136562 | 22.644205 | 680.01897 |
| eqmotion | seed43_neural_cost_moderate | unary | 16 / 645 / 309 | 473 | 0.0013165784 | 0.0039136562 | 22.644205 | 680.01897 |
| eqmotion | seed43_neural_cost_moderate | joint | 16 / 645 / 309 | 473 | 0.0013165784 | 0.0039136562 | 22.644205 | 680.01897 |

Means above weight queries equally. The CSV and JSON also retain the distinct
past-agent-weighted lower bound. Costs use the unchanged past-normalized ADE,
not meters, probabilities or the pending native-unit evaluation proposal.
Full bin counts and observed/exact selected-cost means are in `reliability_bins.csv`.
Empty bins remain empty. These are descriptive cost reliability tables, not ECE.
