# Fixed-Floor Tail-Weighted Risk Results

## Material Passport

Fresh_run:216 Torch risk heads,432,000 updates, frozen inference/decisions, held readout and3,000 locality-bootstrap draws.
Cached_verified: forecasting models, protected damping, ridge utility and input preprocessing.
Twelve development-exposed source localities. No independent confirmation, new forecaster or deployment.

## Prespecified Equal-Count Primary

MSE matched-count minus tail selected positive-harm ratio (percentage points): **undefined**. Positive means the tail model reduces harm.
Matched-count tail ADE gain (%): **0.0033 [-0.0025, 0.0121]**.
Counts match within the same locality/recording/frame, including unknown-label rows; no later-query allocation.

## All Actions

| Action | ADE gain over floor % | ADE gain over CV % | Hard gain over floor % | Easy gain over CV % | Endpoint FDE gain over floor % | Intervention % | Positive harm ratio % |
|---|---:|---:|---:|---:|---:|---:|---:|
| floor | 0.0000 [0.0000, 0.0000] | 0.8259 [0.5401, 1.0971] | 0.0000 [0.0000, 0.0000] | 3.8347 [3.0111, 4.5772] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | undefined |
| mse | 0.1544 [0.1154, 0.1952] | 0.9783 [0.6803, 1.2667] | 0.0948 [0.0406, 0.1645] | 5.4587 [4.2586, 6.5523] | 0.2284 [0.1785, 0.2756] | 7.8187 [6.4940, 9.3360] | 3.1126 [2.5204, 3.6623] |
| mse_matched_count | 0.0637 [0.0444, 0.0836] | 0.8888 [0.5976, 1.1682] | 0.0297 [0.0102, 0.0538] | 4.8034 [3.7825, 5.7206] | 0.0965 [0.0685, 0.1253] | 4.9228 [3.9492, 6.0328] | undefined |
| ridge | 0.5233 [0.3790, 0.6926] | 1.3434 [0.9532, 1.7242] | 0.5379 [0.3339, 0.8055] | 4.3950 [3.3597, 5.3805] | 0.6817 [0.5081, 0.8730] | 5.6345 [4.4486, 6.8915] | 4.9104 [3.8936, 5.9871] |
| tail4 | 0.0670 [0.0448, 0.0925] | 0.8920 [0.6011, 1.1730] | 0.0393 [0.0108, 0.0799] | 4.7853 [3.7709, 5.7025] | 0.1006 [0.0687, 0.1326] | 4.9228 [3.9492, 6.0328] | undefined |

`mse_matched_count` is diagnostic ranking at tail-model counts, not a claim that the MSE predicted budget holds.
Floor intervention is0 by definition of the incremental comparison. Undefined risk is not zero risk.

## Paired Loss Controls

| Tail versus control | ADE gain % | Positive harm reduction pp | Intervention difference pp |
|---|---:|---:|---:|
| mse | -0.0879 [-0.1113, -0.0662] | undefined | -2.8959 [-3.4608, -2.4121] |
| mse_matched_count | 0.0033 [-0.0025, 0.0121] | undefined | 0.0000 [0.0000, 0.0000] |
| ridge | -0.4607 [-0.6320, -0.3126] | undefined | -0.7116 [-2.0719, 0.8518] |

## Worst Views and Label Sensitivity

| Action | Worst easy gain % | Risk-violating views | Undefined-risk views | Zero-CV harmed views | Complete-label ADE gain % | Partial-label ADE gain % |
|---|---:|---:|---:|---:|---:|---:|
| floor | 0.1214 | 0 | 216 | 0 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| mse | -0.0657 | 110 | 0 | 0 | 0.1962 [0.1474, 0.2474] | 0.1096 [0.0723, 0.1524] |
| mse_matched_count | 0.3857 | 94 | 3 | 0 | 0.0850 [0.0621, 0.1064] | 0.0425 [0.0243, 0.0616] |
| ridge | 0.2475 | 171 | 0 | 0 | 0.5989 [0.4438, 0.7661] | 0.4297 [0.2822, 0.6193] |
| tail4 | 0.3857 | 95 | 3 | 0 | 0.0899 [0.0642, 0.1155] | 0.0440 [0.0225, 0.0698] |

| Action | P95 error / floor P95 | Unknown-label interventions per dependent view |
|---|---:|---:|
| floor | 1.0000 [1.0000, 1.0000] | 0.0000 [0.0000, 0.0000] |
| mse | 0.9995 [0.9989, 1.0001] | 23.9676 [5.1156, 52.0293] |
| mse_matched_count | 0.9998 [0.9996, 0.9999] | 15.3935 [2.9346, 35.0344] |
| ridge | 0.9973 [0.9955, 0.9986] | 32.0231 [4.6472, 72.6391] |
| tail4 | 0.9997 [0.9995, 0.9999] | 15.5046 [2.9396, 35.0212] |

Each action has216 dependent held-locality views, not216 independent scenes.
Positive harm is distinct from net ADE and easy degradation. All primary rosters remain fixed.

## Moment Diagnostics

| Loss | Predicted selected harm % | Actual selected harm % | Predicted/actual reference | Zero predicted harm fraction | Harm MSE | Easy harm MSE |
|---|---:|---:|---:|---:|---:|---:|
| mse | 0.7169 [0.6545, 0.7796] | 3.1126 [2.5204, 3.6623] | 3.6947 [3.0490, 4.4767] | 0.0000 [0.0000, 0.0000] | 73.5948 [49.6998, 104.0106] | 1.2454 [0.8463, 1.9076] |
| tail4 | undefined | undefined | undefined | undefined | 93.8657 [64.2175, 131.3183] | 1.7162 [1.1076, 2.4985] |

Moment MSE is in squared image-local ADE units. Weighted loss estimates tilted harm scores, not calibrated means.

## Three Forecaster Seeds

| Seed | MSE gain over floor % | Tail gain over floor % | MSE matched-count gain % | Tail harm ratio % |
|---|---:|---:|---:|---:|
| 17 | 0.1565 [0.1130, 0.2053] | 0.0892 [0.0604, 0.1208] | 0.0788 [0.0536, 0.1019] | undefined |
| 29 | 0.1607 [0.1145, 0.2081] | 0.0628 [0.0360, 0.0970] | 0.0625 [0.0379, 0.0903] | 2.8327 [2.0381, 3.6439] |
| 43 | 0.1460 [0.0979, 0.2038] | 0.0490 [0.0336, 0.0642] | 0.0498 [0.0368, 0.0635] | undefined |

## Gates

- primary_equal_count_harm_reduction: False
- equal_count_ADE_advantage: False
- floor_ADE_advantage: True
- nonzero_each_locality: True
- every_view_easy_preserved: True
- no_zero_CV_harm: True
- every_view_defined_risk_within_budget: False
- exploratory_joint_screen_pass: False
- calibration_certificate: False
- independent_confirmation: False
- deployment_changed: False
- stage5c_executed: False
- smc_enabled: False

Source and seed breakdowns are retained for every metric in summary.json. Bootstrap unit:12 locality means after averaging dependent seed/producer/fit-half views. Overlapping windows are not independent.
All intervals are unadjusted development uncertainty, not a safety certificate or final independent test.
Image-local detector silver, obs8/pred12 rawstride12; not historicalt50, metric, seconds, human gold, physical safety, true3D or foundation.
Stage5C and SMC remain disabled. The research goal is not complete.
