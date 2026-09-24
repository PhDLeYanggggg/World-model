# New Forest Fitting Losses

Twelve full-EqMotion forests were newly fitted; no forecasting or neural cost model was retrained.
Each fixed 128-tree fit uses its paired neural head's actual 768,000 sample draws.
The loss is draw-weighted mean squared error of benefit/harm fractions on fitting rows
with positive forecast disagreement. It is not validation loss, ADE, or a convergence claim.

| View | Trees | First recorded MSE (16 trees) | Final MSE (128 trees) | Effective rows | Fit seconds |
|---|---:|---:|---:|---:|---:|
| coupa_seed17 | 128 | 0.09822813 | 0.09794417 | 116450 | 28.824 |
| coupa_seed29 | 128 | 0.09799563 | 0.09764972 | 116325 | 29.034 |
| coupa_seed43 | 128 | 0.09839273 | 0.09819836 | 116465 | 29.238 |
| deathCircle_seed17 | 128 | 0.09522079 | 0.09459805 | 113875 | 27.753 |
| deathCircle_seed29 | 128 | 0.09481173 | 0.09421841 | 113746 | 27.461 |
| deathCircle_seed43 | 128 | 0.09594969 | 0.09536864 | 113888 | 27.782 |
| gates_seed17 | 128 | 0.09617838 | 0.09571611 | 125985 | 31.054 |
| gates_seed29 | 128 | 0.09700555 | 0.09622347 | 125856 | 31.509 |
| gates_seed43 | 128 | 0.09670920 | 0.09646292 | 125996 | 37.275 |
| hyang_seed17 | 128 | 0.09294344 | 0.09303072 | 67486 | 20.956 |
| hyang_seed29 | 128 | 0.09354390 | 0.09309546 | 67486 | 13.195 |
| hyang_seed43 | 128 | 0.09441363 | 0.09366470 | 67484 | 13.353 |

Summed fitting-loop time: 317.433 seconds. This excludes loading,
between-fit overhead and wall-clock interruptions. Full 16-tree traces remain in analysis.json.
All budgets completed without sample reduction. Lower fitting loss alone does not establish safer selection.
