# Absolute Cost Context

Post-hoc descriptive context; no model selection or replacement gate.
All864 locality/method/subset rows are retained in absolute_costs.csv. Unknown fields stay empty.
Means average three seeds within locality. Repeated assignments are dependent.
Coverage is the mean of seed-specific ratios, not the ratio of pooled means.
Large relative losses can result from small comparator MSE; the registered ratios remain unchanged.
These expected-cost squared errors are not trajectory FDE/ADE or physical-safety measurements.

| Family/method | MSE min / median / max across24 dependent locality views | Missing |
|---|---|---:|
| full/cap_aux_mass | [0.00448906354614281, 0.05391018559924915, 1.2714018850632751] | 0 |
| full/cap_aux_raw | [0.0035695233837801424, 0.04501189625921223, 1.2764270992983973] | 0 |
| full/cap_aux_scaled | [0.003465265708032518, 0.040714573968572546, 1.2787262030518494] | 0 |
| full/cost_only_mass | [0.004454644854098486, 0.054298058195014605, 1.270054992165863] | 0 |
| full/cost_only_raw | [0.00354801440147982, 0.0423784581107004, 1.275855692517396] | 0 |
| full/cost_only_scaled | [0.003466430036462612, 0.04100483472230834, 1.278876871809446] | 0 |
| full/shuffled_aux_mass | [0.004924599146439608, 0.052873826552902875, 1.269534822668818] | 0 |
| full/shuffled_aux_raw | [0.0035632353075267247, 0.044522991031407826, 1.2759376920906045] | 0 |
| full/shuffled_aux_scaled | [0.0034725707009181893, 0.04127643088954269, 1.2788672432946393] | 0 |
| motion_only/cap_aux_mass | [0.0005398562084835607, 0.047260470474573615, 1.064889362181008] | 0 |
| motion_only/cap_aux_raw | [0.0002012444176080193, 0.029899202369689425, 1.067338595555844] | 0 |
| motion_only/cap_aux_scaled | [1.6252489373047514e-05, 0.01436815312333218, 1.0673351691232196] | 0 |
| motion_only/cost_only_mass | [0.0003146925035246378, 0.04726782289075639, 1.0651036419205189] | 0 |
| motion_only/cost_only_raw | [0.00016522820792984412, 0.030136694427959666, 1.067284919271432] | 0 |
| motion_only/cost_only_scaled | [1.0234834022645345e-05, 0.014357313336493364, 1.0674912233973695] | 0 |
| motion_only/shuffled_aux_mass | [0.0003405290113846085, 0.047136845747870304, 1.0648361903494656] | 0 |
| motion_only/shuffled_aux_raw | [0.00016070399505792042, 0.0298654342751742, 1.0672382472398045] | 0 |
| motion_only/shuffled_aux_scaled | [7.84261373059957e-06, 0.014358918592464073, 1.06735474433602] | 0 |
