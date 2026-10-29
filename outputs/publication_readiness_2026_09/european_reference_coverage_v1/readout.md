# Causal Reference-Coverage Readout

All nine endpoints fixed before scoring. Source development only, not independent confirmation.
Six unchanged-radius fits are explicit neutral controls; repeated seed/context errors are averaged within locality.

| Endpoint | Harm-aware control vs CV (%) | Coverage vs CV (%) | Primary change (pp) | Conditional 95% CI (pp) | Change vs original ADE-only (pp) |
|---|---:|---:|---:|---|---:|
| ADE_all | 7.531916 | 7.595475 | 0.063559 | [0.02992344904626709, 0.09663506945717852] | -0.866165 |
| ADE_complete | 11.053453 | 11.130813 | 0.077360 | [0.02528294308879639, 0.13251613110585453] | -1.174787 |
| ADE_easy | -8.433059 | -8.345904 | 0.087155 | [0.026951050636038608, 0.15717294346170171] | 2.823304 |
| ADE_hard | 11.675472 | 11.734517 | 0.059045 | [0.025373774944571086, 0.0929194221907911] | -1.620695 |
| FDE_all | 11.588928 | 11.666115 | 0.077187 | [0.03213198665476756, 0.12104697965448356] | -1.732230 |
| FDE_complete | 12.717251 | 12.800247 | 0.082996 | [0.022426746471648492, 0.1473923373149072] | -1.545067 |
| FDE_easy | -1.392642 | -1.244353 | 0.148289 | [0.07058437894714246, 0.23255295657171945] | 2.594493 |
| FDE_hard | 16.065862 | 16.131904 | 0.066042 | [0.020569815319996364, 0.1119822343133672] | -1.781240 |

Original comparator metric checks: 2304; every fixed endpoint/locality/subset retained.
All 163 recordings, including missing-label and zero-reference cases, remain in the diagnostic CSV.
Known positive harm is separate from net error. Forecast diagnostics cannot certify the complete policy.
Changed forecasts invalidate inherited candidate-dependent safety heads. No deployment promotion.
Image pixels/raw frames and released detector tracks only; Stage5C/SMC remain off.
