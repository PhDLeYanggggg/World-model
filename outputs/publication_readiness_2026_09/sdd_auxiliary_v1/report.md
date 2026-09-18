# Matched SDD Auxiliary Training

54 real fits; three seeds, three exposed fit scenes. No independent confirmation or deployment.

| Source / modality | Gain vs CV (%) | Gain vs no-aux same input (%) | Gain vs same-source mask (%) | Safe positive fits |
| --- | ---: | ---: | ---: | ---: |
| no_aux_geometry | -1.34976 | 0.00000 | 0.02236 | 0/9 |
| no_aux_mask_only | -1.37243 | 0.00000 | 0.00000 | 0/9 |
| no_aux_past_rgb | -1.98266 | 0.00000 | -0.60197 | 0/9 |
| sdd_aux_geometry | -0.80523 | 0.53728 | 0.01094 | 0/9 |
| sdd_aux_mask_only | -0.81626 | 0.54864 | 0.00000 | 0/9 |
| sdd_aux_past_rgb | -1.27335 | 0.69552 | -0.45339 | 0/9 |

All outputs and negative fits retained. Bootstrap is conditional on three previously exposed physical sites.
Partial auxiliary labels never change the complete-label main primary. No metric/seconds claim.
