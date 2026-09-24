# Risk-Head Training Loss

36 fresh forests, six bounded targets, same fixed128-tree budget.
Weighted fitting MSE is not independent validation or a safety guarantee.

| View | Action | Trees | Fitting MSE | Fitting seconds | Unknown draws |
|---|---|---:|---:|---:|---:|
| coupa_seed17 | damped_velocity_005 | 128 | 0.0593468 | 23.321 | 0 |
| coupa_seed17 | transformer | 128 | 0.0610258 | 26.822 | 0 |
| coupa_seed17 | eqmotion | 128 | 0.0621205 | 29.565 | 0 |
| coupa_seed29 | damped_velocity_005 | 128 | 0.0594442 | 23.866 | 0 |
| coupa_seed29 | transformer | 128 | 0.0619754 | 26.586 | 0 |
| coupa_seed29 | eqmotion | 128 | 0.0620017 | 29.592 | 0 |
| coupa_seed43 | damped_velocity_005 | 128 | 0.0594145 | 23.625 | 0 |
| coupa_seed43 | transformer | 128 | 0.0613450 | 27.085 | 0 |
| coupa_seed43 | eqmotion | 128 | 0.0622048 | 29.723 | 0 |
| deathCircle_seed17 | damped_velocity_005 | 128 | 0.0509465 | 21.193 | 0 |
| deathCircle_seed17 | transformer | 128 | 0.0557606 | 25.906 | 0 |
| deathCircle_seed17 | eqmotion | 128 | 0.0578345 | 28.278 | 0 |
| deathCircle_seed29 | damped_velocity_005 | 128 | 0.0506254 | 21.173 | 0 |
| deathCircle_seed29 | transformer | 128 | 0.0554249 | 26.300 | 0 |
| deathCircle_seed29 | eqmotion | 128 | 0.0573622 | 28.468 | 0 |
| deathCircle_seed43 | damped_velocity_005 | 128 | 0.0505247 | 20.895 | 0 |
| deathCircle_seed43 | transformer | 128 | 0.0555593 | 26.451 | 0 |
| deathCircle_seed43 | eqmotion | 128 | 0.0581087 | 28.652 | 0 |
| gates_seed17 | damped_velocity_005 | 128 | 0.0536529 | 24.166 | 0 |
| gates_seed17 | transformer | 128 | 0.0579450 | 28.546 | 0 |
| gates_seed17 | eqmotion | 128 | 0.0596048 | 32.180 | 0 |
| gates_seed29 | damped_velocity_005 | 128 | 0.0533068 | 23.928 | 0 |
| gates_seed29 | transformer | 128 | 0.0574039 | 28.570 | 0 |
| gates_seed29 | eqmotion | 128 | 0.0597318 | 32.267 | 0 |
| gates_seed43 | damped_velocity_005 | 128 | 0.0534332 | 24.075 | 0 |
| gates_seed43 | transformer | 128 | 0.0580155 | 28.729 | 0 |
| gates_seed43 | eqmotion | 128 | 0.0597764 | 32.488 | 0 |
| hyang_seed17 | damped_velocity_005 | 128 | 0.0496289 | 9.346 | 0 |
| hyang_seed17 | transformer | 128 | 0.0552386 | 12.173 | 0 |
| hyang_seed17 | eqmotion | 128 | 0.0561060 | 14.065 | 0 |
| hyang_seed29 | damped_velocity_005 | 128 | 0.0493133 | 9.320 | 0 |
| hyang_seed29 | transformer | 128 | 0.0549844 | 12.067 | 0 |
| hyang_seed29 | eqmotion | 128 | 0.0557802 | 13.785 | 0 |
| hyang_seed43 | damped_velocity_005 | 128 | 0.0495114 | 9.411 | 0 |
| hyang_seed43 | transformer | 128 | 0.0552766 | 11.915 | 0 |
| hyang_seed43 | eqmotion | 128 | 0.0564082 | 14.001 | 0 |

Total fitting-loop seconds: 828.5308061260002.
This excludes preparation, solving, hashing and replay. All six loss traces and held-source target MSE remain in analysis.json.

## Matched Held-Source Errors

Equal mean across the 12 excluded-site/seed views per action. These are already design-exposed source sites, not independent validation.
Only complete-label rows enter these target errors; the main decision population still includes partial and unknown futures.
Targets, complete-label counts and source views are paired across all three representations. Lower MSE does not certify calibration.

| Action | Representation | Overall benefit | Overall harm | Easy harm | Easy benefit | Easy denominator | Easy probability |
|---|---|---:|---:|---:|---:|---:|---:|
| damped_velocity_005 | native | 0.1668396 | 0.0824342 | 0.0290787 | 0.0192825 | 0.0433133 | 0.1758868 |
| damped_velocity_005 | dimensionless | 0.1763709 | 0.0827071 | 0.0290832 | 0.0145702 | 0.0450494 | 0.2069470 |
| damped_velocity_005 | cutoff_relative | 0.1696570 | 0.0823060 | 0.0290603 | 0.0194180 | 0.0434862 | 0.1774812 |
| transformer | native | 0.0967526 | 0.1176617 | 0.0681125 | 0.0175894 | 0.0431717 | 0.0771038 |
| transformer | dimensionless | 0.0974154 | 0.1323703 | 0.0797787 | 0.0185390 | 0.0435745 | 0.0818681 |
| transformer | cutoff_relative | 0.0966903 | 0.1177591 | 0.0681777 | 0.0175498 | 0.0431755 | 0.0770610 |
| eqmotion | native | 0.1109496 | 0.1034417 | 0.0540426 | 0.0169549 | 0.0405794 | 0.0731649 |
| eqmotion | dimensionless | 0.1123863 | 0.1053227 | 0.0551922 | 0.0187853 | 0.0427266 | 0.0787221 |
| eqmotion | cutoff_relative | 0.1109382 | 0.1033932 | 0.0540258 | 0.0170081 | 0.0406017 | 0.0732924 |