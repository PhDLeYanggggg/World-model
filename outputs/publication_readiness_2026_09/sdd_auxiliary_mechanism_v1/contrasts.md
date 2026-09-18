# Source Mechanism Contrasts

All 108 fixed results are retained: 54 fresh fits and 54 hash-verified previous fits.
No model or threshold is selected here. Three reused sites do not provide independent confirmation.

| Input / contrast | Gain (%) | Descriptive site CI (%) | Better seed-site fits |
| --- | ---: | --- | ---: |
| geometry:sdd_aux_vs_main4k | 0.22356 | [-0.10449, 2.54107] | 6/9 |
| geometry:sdd_permuted_vs_main4k | 0.10579 | [0.02197, 0.65654] | 6/9 |
| geometry:sdd_aux_vs_sdd_permuted | 0.11789 | [-0.12649, 1.89699] | 6/9 |
| geometry:no_aux_vs_main4k | -0.31542 | [-2.60490, -0.04449] | 1/9 |
| geometry:sdd_aux_vs_no_aux | 0.53728 | [-0.05998, 5.01533] | 7/9 |
| mask_only:sdd_aux_vs_main4k | 0.18465 | [-0.12188, 2.21882] | 6/9 |
| mask_only:sdd_permuted_vs_main4k | 0.19997 | [0.03173, 2.06873] | 7/9 |
| mask_only:sdd_aux_vs_sdd_permuted | -0.01536 | [-0.15366, 0.15464] | 5/9 |
| mask_only:no_aux_vs_main4k | -0.36600 | [-2.83639, -0.06170] | 1/9 |
| mask_only:sdd_aux_vs_no_aux | 0.54864 | [-0.06014, 4.91578] | 6/9 |
| past_rgb:sdd_aux_vs_main4k | 0.24933 | [0.06632, 1.95412] | 6/9 |
| past_rgb:sdd_permuted_vs_main4k | 0.19117 | [-0.11281, 1.68656] | 7/9 |
| past_rgb:sdd_aux_vs_sdd_permuted | 0.05827 | [-0.17034, 0.27215] | 6/9 |
| past_rgb:no_aux_vs_main4k | -0.44932 | [-0.76734, -0.32033] | 1/9 |
| past_rgb:sdd_aux_vs_no_aux | 0.69552 | [0.38541, 2.70073] | 9/9 |

Positive gain is relative to the named neural control, not necessarily to causal CV.
main4k matches main-training exposure, not total compute.
Source permutation matches draws/compute and preserves recording/support strata; it does not remove all dependence.
The absolute-ADE decomposition in analysis.json is arithmetic, not a causal mediation result.
All event strata use evaluation labels only. They are not inference inputs.
No metric, seconds-level, true-3D, foundation, Stage5C or SMC claim.
