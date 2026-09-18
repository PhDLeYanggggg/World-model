# Training-Side Candidate Cross-Fitting

## Material Passport

Twelve cold-start models, three seeds, four inner source sites, 120,000 updates.
Bookstore and all main roles excluded from every fit and inference.
OOF targets are training supervision, not independent confirmation or a deployed risk head.

| Inner held site | Rows | Mean training gain (%) | OOF gain (%) | Conditional recording 95% CI | Easy pixel harm |
| --- | ---: | ---: | ---: | --- | ---: |
| coupa | 4250 | +0.795080 | -2.193170 | [-3.3252422842239997, -1.7228640456008504] | 0.05345731 |
| deathCircle | 3054 | +1.082096 | -2.764347 | [-3.542617558080906, -1.330960888172004] | 0.04873619 |
| gates | 1662 | +0.544019 | -7.758925 | [-13.367769352504393, -3.1613714031021143] | 0.07161446 |
| hyang | 6464 | +2.543802 | -8.810146 | [-17.592572173707765, -4.903661676789843] | 0.14254284 |

Primary equal-site gain: -5.015980%.
Conditional four-site interval: [-8.396553368897642, -2.487726531034684].
Window-weighted sensitivity: -5.456920%.
Per-seed window gains: [-5.611063116329862, -6.079791600687079, -4.679905975156795].
Mean per-seed binary future-oracle gain: 0.527071% (not a policy).

The cached full-four-site in-sample reference is descriptive only. Smaller inner
training sets, site shift and normalization change prevent a causal attribution
of the entire gap to training-error optimism. No favorable site/seed is selected.
Bootstrap uncertainty is conditional on four explored sites and overlapping fits.
Easy percentage is undefined, not a safety pass. Annotation pixels/past-normalized
raw-frame task only; no metric/seconds/foundation claim. Stage5C/SMC remain off.
