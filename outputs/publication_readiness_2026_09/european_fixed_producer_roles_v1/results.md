# Fixed Four-Source Producer Roles: Complete Development Results

I held the four-source candidate and protected fallback fixed and separated producer fitting A, controller supervision B, and readout C.
Six ordered source rotations, three seeds and two event targets give36 dependent groups. All144 Torch heads reached2,000 updates;72 fixed-alpha ridge heads were also fitted.
All180 decisions froze before new comparison readout. No trajectory forecaster was retrained. Reserved model-selection/calibration/confirmation roles remain closed.

## Same-Forecast Primary Comparisons

| Matched supervision vs | All ADE gain (%) | Positive / negative CI | Hard gain (%) | Complete gain (%) | Endpoint FDE gain (%) |
|---|---:|---:|---:|---:|---:|
| oof_control | -0.9801 to 1.1255 | 21 / 6 | -0.9216 to 1.0817 | -1.5589 to 1.5158 | -1.0705 to 1.6342 |
| ridge | -3.0845 to 2.7663 | 14 / 15 | -3.7770 to 2.4358 | -3.9514 to 4.0313 | -4.9631 to 5.5881 |
| old_stop | -0.9007 to 1.6471 | 12 / 3 | -0.8374 to 1.9401 | -1.2393 to 2.1474 | -1.3925 to 2.2213 |

## Accuracy And Safety

| Policy | All gain vs floor4 (%) | All gain vs old stop (%) | Positive / negative CI vs old stop | Worst easy degradation vs CV (%) | Zero-CV harmed views | Switch rate |
|---|---:|---:|---:|---:|---:|---:|
| producer_matched | 0.0614 to 2.5676 | -0.9007 to 1.6471 | 12 / 3 | 0.23767 | 0 | 0.0047 to 0.5508 |
| oof_control | 0.0428 to 1.8755 | -1.5454 to 1.5651 | 11 / 17 | 7.85031 | 0 | 0.0127 to 0.3566 |
| ridge | -1.7018 to 3.0451 | -2.8922 to 2.9089 | 15 / 13 | 8.23348 | 0 | 0.0282 to 0.2299 |
| old_stop | 0.0914 to 1.7825 | 0 to 0 | 0 / 0 | 0.00000 | 0 | 0.0037 to 0.3488 |
| raw_neural | -19.8510 to 11.0111 | -21.5626 to 10.8546 | 24 / 6 | 81.29545 | 12 | 1.0000 to 1.0000 |

Raw neural is unprotected diagnostic, not a candidate. Positive-harm ratios below are different from net easy degradation.

| Policy | Supported selected locality/views | Above2% realized harm ratio | Underpredicted | Realized ratio range | Predicted ratio range |
|---|---:|---:|---:|---:|---:|
| producer_matched | 144 | 66 | 122 | 0.0000 to 0.1517 | 0.0028 to 0.0127 |
| oof_control | 144 | 69 | 108 | 0.0010 to 0.2536 | 0.0023 to 0.0120 |
| ridge | 144 | 111 | 136 | 0.0000 to 1.2455 | 0.0001 to 0.0097 |
| old_stop | 144 | 69 | 120 | 0.0000 to 0.1047 | 0.0018 to 0.0128 |

## Easy Failure Accounting

| Policy | Violations | Already-bad floor views | New violations from controller | Violations despite helpful increment |
|---|---:|---:|---:|---:|
| producer_matched | 0 | 0 | 0 | 0 |
| oof_control | 6 | 0 | 6 | 0 |
| ridge | 5 | 0 | 5 | 0 |
| old_stop | 0 | 0 | 0 | 0 |
| raw_neural | 68 | 0 | 68 | 0 |

Every rotation, event, seed and source locality is retained in groups/*.json and *_localities.csv, including p95/p99 tails.
Four-locality bootstraps use3,000 paired resamples per view, not independent overlapping windows. Views share source data and are not independent tests; no multiplicity-adjusted confirmatory claim is made.
The matched/OOF contrast changes training producer, source count, rollout quality, labels and fitted feature statistics jointly. It is not isolated producer-identity causality.
Both arms share supervised draw arrays, masks, source weights, CV-cost scales, capacity and budget. Their target labels, feature statistics and fixed ranking scales legitimately differ.
The original A-derived easy/hard cutoffs are common. B supplies all new head supervision/normalization; C supplies no fitted statistics.
Image-pixel obs8/pred12 at raw annotation stride12, EuropeanSquares released detector tracks. No metric/seconds, human gold, physical safety, true3D or foundation claim.
Partial-window ADE and actual endpoint FDE are separate. Unknown future labels are not zero errors. No deployment change, Stage5C or SMC.
