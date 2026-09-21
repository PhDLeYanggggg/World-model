# Complete Cost-Head Fit Results

Post-hoc in-sample diagnosis. Each head uses the same 11,966 query identities
and 306 causal/predicted-rollout features. OOF refers to the trajectory
producer; these rows trained the cost head itself. No independent calibration,
test selection, deployment or new fit. All costs retain past-normalized ADE.

## Whole Fit Population

| Family | Seed | Head | True mean harm | Predicted mean harm | Harm MSE | Mean-label reference MSE |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| transformer | 17 | ridge | 0.84152909 | 0.86514221 | 0.96693999 | 39.438056 |
| transformer | 17 | neural_cost | 0.84152909 | 0.84556683 | 1.1806853 | 39.438056 |
| transformer | 29 | ridge | 0.59695006 | 0.61563763 | 0.37593129 | 44.437289 |
| transformer | 29 | neural_cost | 0.59695006 | 0.60062105 | 1.8664382 | 44.437289 |
| transformer | 43 | ridge | 0.87367696 | 0.89442368 | 0.78021121 | 54.314313 |
| transformer | 43 | neural_cost | 0.87367696 | 0.91076536 | 0.95085579 | 54.314313 |
| eqmotion | 17 | ridge | 3.6394336 | 3.8624886 | 32.525808 | 248.93337 |
| eqmotion | 17 | neural_cost | 3.6394336 | 3.5541557 | 38.747608 | 248.93337 |
| eqmotion | 29 | ridge | 2.9314322 | 3.0739625 | 24.596859 | 354.03683 |
| eqmotion | 29 | neural_cost | 2.9314322 | 2.6597001 | 116.17171 | 354.03683 |
| eqmotion | 43 | ridge | 5.5844023 | 5.6622804 | 20.328869 | 347.85248 |
| eqmotion | 43 | neural_cost | 5.5844023 | 5.4478359 | 69.618585 | 347.85248 |

## Fixed Per-Agent Eligibility

Conservative: predicted gain >=0.02 and predicted harm <=0.05. Moderate:
gain >=0.01 and harm <=0.1. These are original rules, not searched thresholds.
Eligibility is not a scene-solver decision or intervention rate. Positive gain
means benefit minus harm. All nonempty groups are retained, including the small
positive EqMotion seed17/neural/conservative result.

| Family | Seed | Head | Rule | Rows | True harm | Predicted harm | True net gain | Predicted net gain |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| transformer | 17 | ridge | conservative | 2055 | 0.11120122 | 0.00392554 | -0.094792029 | 0.10845047 |
| transformer | 17 | ridge | moderate | 2381 | 0.1089908 | 0.0075235905 | -0.091992486 | 0.098202904 |
| transformer | 17 | neural_cost | conservative | 2299 | 0.086396302 | 0.018590356 | -0.039603641 | 0.067919358 |
| transformer | 17 | neural_cost | moderate | 2891 | 0.09363706 | 0.022351535 | -0.039655376 | 0.072803564 |
| transformer | 29 | ridge | conservative | 1440 | 0.036065134 | 0.0015861061 | -0.0072631971 | 0.053923 |
| transformer | 29 | ridge | moderate | 1888 | 0.034662464 | 0.0026031617 | -0.0088701829 | 0.044809504 |
| transformer | 29 | neural_cost | conservative | 285 | 0.055643624 | 0.014273152 | -0.011064841 | 0.041945319 |
| transformer | 29 | neural_cost | moderate | 575 | 0.059674973 | 0.020637451 | -0.010107087 | 0.038383109 |
| transformer | 43 | ridge | conservative | 562 | 0.33493893 | 0.00058605365 | -0.31011405 | 0.046501023 |
| transformer | 43 | ridge | moderate | 907 | 0.29020728 | 0.0012330903 | -0.27075968 | 0.034470013 |
| transformer | 43 | neural_cost | conservative | 330 | 0.058482495 | 0.016518561 | -0.023751277 | 0.07174423 |
| transformer | 43 | neural_cost | moderate | 519 | 0.074842645 | 0.02857656 | -0.040525411 | 0.062323193 |
| eqmotion | 17 | ridge | conservative | 1965 | 0.30251389 | 0.00045982135 | -0.28082956 | 0.096946057 |
| eqmotion | 17 | ridge | moderate | 2534 | 0.25130548 | 0.0014990409 | -0.23209559 | 0.079594666 |
| eqmotion | 17 | neural_cost | conservative | 44 | 0.063567878 | 0.026807466 | 0.00094117713 | 0.061120962 |
| eqmotion | 17 | neural_cost | moderate | 118 | 0.067399828 | 0.050689444 | -0.024546246 | 0.06654561 |
| eqmotion | 29 | ridge | conservative | 3067 | 0.15101822 | 0.00071400192 | -0.13096049 | 0.069887168 |
| eqmotion | 29 | ridge | moderate | 3539 | 0.15875962 | 0.0016797599 | -0.14014877 | 0.06372551 |
| eqmotion | 29 | neural_cost | conservative | 170 | 0.89219692 | 0.021514019 | -0.85240312 | 0.098095231 |
| eqmotion | 29 | neural_cost | moderate | 329 | 0.71842514 | 0.037553141 | -0.68478193 | 0.08845774 |
| eqmotion | 43 | ridge | conservative | 1640 | 0.075864597 | 0.00065340036 | -0.056195572 | 0.074557193 |
| eqmotion | 43 | ridge | moderate | 1803 | 0.073388234 | 0.0017358851 | -0.05440546 | 0.06954594 |
| eqmotion | 43 | neural_cost | conservative | 186 | 0.24318398 | 0.021308845 | -0.21234268 | 0.093891342 |
| eqmotion | 43 | neural_cost | moderate | 272 | 0.20624663 | 0.032903707 | -0.17283108 | 0.096006069 |

## Ridge Projection And Target Concentration

| Family | Seed | Negative raw harm rows | Share (%) | Actually harmed among these (%) | Mean actual harm | Top 1% squared harm-label mass (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| transformer | 17 | 2358 | 19.70583 | 69.59288 | 0.092835395 | 91.88774 |
| transformer | 29 | 2475 | 20.68360 | 65.37374 | 0.034734735 | 98.72277 |
| transformer | 43 | 1266 | 10.57998 | 74.32859 | 0.28198588 | 96.58555 |
| eqmotion | 17 | 6073 | 50.75213 | 73.76914 | 0.1647999 | 79.52011 |
| eqmotion | 29 | 4382 | 36.62042 | 68.21086 | 0.15531907 | 91.60678 |
| eqmotion | 43 | 2288 | 19.12084 | 71.41608 | 0.075144248 | 56.23267 |

The six target populations are shared by their ridge/neural heads. The largest
1% contains 120 rows per population. Squared label mass is not the final loss
contribution, and does not by itself prove a causal training defect. Projection
of a negative value to zero increases it; removing projection is not a remedy.
Per-recording results and empty subsets are retained in `all_fit_slices.csv`.
