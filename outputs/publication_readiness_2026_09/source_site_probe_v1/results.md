# Source-Site Visual Information Results

Internal held-physical-site diagnostic of original SDD train40 only. No formal main split changed.
All five sites and three seeds are retained. Average seed losses, not ensemble predictions.
Positive absolute Brier reduction is not percentage ADE/FDE improvement.

Primary equal-site RGB-minus-mask lift: -0.020046; conditional site-block interval [-0.027587, -0.011995].
Only five physical sites, overlapping training folds and already exposed source data: not independent confirmation.

| Held site | Windows / agents / videos | RGB Brier | Lift vs mask | Video-block conditional95% interval | Lift vs training prior | Positive seeds |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| bookstore | 6944 / 181 / 7 | 0.296312 | -0.021543 | [-0.047856, -0.003369] | -0.047988 | 0/3 |
| coupa | 4250 / 79 / 4 | 0.314085 | -0.031010 | [-0.041552, +0.008300] | -0.039439 | 0/3 |
| deathCircle | 3054 / 177 / 5 | 0.327462 | -0.028462 | [-0.037883, -0.010889] | -0.087600 | 0/3 |
| gates | 1662 / 78 / 7 | 0.256285 | -0.008525 | [-0.028679, +0.009638] | -0.022617 | 1/3 |
| hyang | 6464 / 211 / 13 | 0.293818 | -0.010692 | [-0.028987, +0.011596] | -0.046279 | 0/3 |

## Weighting and Prior Sensitivity

| Site | Equal-agent RGB lift [conditional95% interval] | Mask mean-shift term | Mask varying-prediction term | Prior varying-prediction term | RGB AUROC | RGB ECE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| bookstore | -0.028716 [-0.042169, -0.015162] | -0.003678 | -0.017865 | -0.044315 | 0.495691 | 0.180041 |
| coupa | -0.019249 [-0.041494, +0.000955] | -0.025215 | -0.005795 | -0.013445 | 0.567109 | 0.259674 |
| deathCircle | -0.028086 [-0.037135, -0.018605] | -0.027488 | -0.000973 | -0.036411 | 0.458361 | 0.276832 |
| gates | -0.008167 [-0.026037, +0.009613] | -0.009699 | +0.001174 | -0.009381 | 0.589524 | 0.243522 |
| hyang | +0.000773 [-0.011165, +0.013726] | -0.006919 | -0.003773 | -0.038823 | 0.496979 | 0.188260 |

Equal-site training-prior lift: -0.048785; equal-site/equal-agent mask lift: -0.016689.
Agent-weighted contrasts answer a different question and do not replace the primary window-within-site contrast.
Brier decomposition uses held labels for fixed-result diagnosis only; no held-label calibration is applied.
The target is any annotation-coordinate change, not human motion intention. Incomplete future labels remain unscored.
SDD8-to12 at stride12 is +144rawframes, not physically time-equated to main. No metric/seconds claim.
No forecast policy, model promotion, Stage5C or SMC.
