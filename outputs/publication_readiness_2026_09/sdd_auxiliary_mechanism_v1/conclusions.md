# Auxiliary Source Mechanism: Complete Fixed Comparison

Result source: 54 fresh Torch fits, 54 cached_verified controls. No independent confirmation.
Main cohort: all 11,966 fit windows; primary: past-normalized ADE, equal physical scene.
No held-score model selection, no sealed-role access and no deployment.

| Schedule / input | Gain vs CV (%) | Gain vs main4k (%) | Gain vs shuffled source (%) | Pixel gain vs mask (%) | Safe positive fits |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_aux_geometry | -1.34976 | -0.31542 | -0.42166 | 0.02236 | 0/9 |
| no_aux_mask_only | -1.37243 | -0.36600 | -0.56711 | 0.00000 | 0/9 |
| no_aux_past_rgb | -1.98266 | -0.44932 | -0.64171 | -0.60197 | 0/9 |
| sdd_aux_geometry | -0.80523 | 0.22356 | 0.11789 | 0.01094 | 0/9 |
| sdd_aux_mask_only | -0.81626 | 0.18465 | -0.01536 | 0.00000 | 0/9 |
| sdd_aux_past_rgb | -1.27335 | 0.24933 | 0.05827 | -0.45339 | 0/9 |
| main4k_geometry | -1.03110 | 0.00000 | -0.10591 | -0.02806 | 0/9 |
| main4k_mask_only | -1.00276 | 0.00000 | -0.20037 | 0.00000 | 0/9 |
| main4k_past_rgb | -1.52648 | 0.00000 | -0.19153 | -0.51853 | 0/9 |
| sdd_permuted_geometry | -0.92421 | 0.10579 | 0.00000 | -0.12245 | 0/9 |
| sdd_permuted_mask_only | -0.80078 | 0.19997 | 0.00000 | 0.00000 | 0/9 |
| sdd_permuted_past_rgb | -1.33240 | 0.19117 | 0.00000 | -0.52740 | 0/9 |

All seed/site failures and absolute easy harm are retained in report.json and fit_metrics.csv.
The 2,000-draw site intervals resample only three already exposed physical sites.
main4k matches main exposure, not total updates; source permutation matches source draws and total updates.
The label ablation preserves recording/support strata, not every dependency or semantic property.
No metric, seconds, true-3D, foundation, Stage5C or SMC claim.
