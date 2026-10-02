# Past Quality Helps Prediction, Not Safe Selection

## Result

All 144 registered auxiliary models completed, with exact paired refitting,
serialized inference and readout replay. The seven fixed past-quality features
have incremental predictive value in this exposed-development experiment.
However, the original 2% selected-harm screen deteriorates. **Do not advance this
arm to transfer or deployment.** No independent result or CVPR-readiness claim.

The comparison keeps 72 frozen forests, their branches, targets, source
partitions, thresholds and original train-only scaling. It adds centered ridge
slopes in each leaf. The placebo permutes the same seven features jointly within
each training recording. Both arms use actual past features at inference.
This is cost-model training, not new neural dynamics or new trajectory forecasts.

## Registered Contrasts

Differences are quality minus the named comparator. Lower MSE is better; higher
utility is better. Utility is conservative selected net gain divided by full
known reference mass, expressed in percent, **not an ADE/FDE improvement**.
Repeated heads/controllers are averaged within locality before 3,000 bootstrap
draws over 12 localities. All intervals are nominal, exposed-development only.

| Comparator | Signed-score MSE change [95% CI] | Full utility change, % [95% CI] | Query-count-matched utility change, % [95% CI] |
|---|---:|---:|---:|
| Original forest | -0.035493 [-0.067255, -0.011377] | +0.287244 [0.126543, 0.490630] | +0.061773 [0.020429, 0.107359] |
| Shuffled-feature placebo | -0.048875 [-0.086578, -0.016930] | +0.240063 [0.080971, 0.451970] | +0.029512 [0.001628, 0.067713] |

All six registered predictive/utility contrasts support this specific information
arm on development data. They do not overcome its three failed safety screens.

## Safety

| Policy | Selected occurrences | Unknown selected | Complete support /72 | Defined easy-risk views | Upper violations | Known-label violations |
|---|---:|---:|---:|---:|---:|---:|
| Original | 95,455 | 918 | 33 | 43 | 7 | 4 |
| Quality | 112,433 | 1,143 | 21 | 62 | 41 | 20 |
| Placebo | 98,083 | 953 | 33 | 49 | 11 | 4 |
| Original matched to quality | 91,961 | 875 | 32 | 41 | 6 | 3 |
| Quality matched to original | 91,961 | 903 | 17 | 41 | 21 | 3 |

Counts repeat rows across heads/controllers; they are not independent people or
scenes. Unknown outcomes remain unknown. Empty or undefined risk views fail.
Quality loses 20 formerly supported groups and gains eight, leaving a net loss
of 12. Its three head seeds have 14, 13 and 14 upper violations respectively;
discarding one seed would not repair the result and is not allowed.

The worst quality upper ratio is 10.3081 (1030.81%), versus the original 0.05406
(5.406%). This is an unknown-outcome completion bound, not observed error or a
probability. Nine quality violations have no selected unknown outcomes; the
failure therefore cannot be dismissed as missing-label conservatism alone.
See [failure decomposition](failure_analysis.md) for numerator/denominator details.

## Evidence Status

- `fresh_run`: 144 train-only ridge fits, exact refits, validation predictions,
  matched controls, bootstrap, remote checkpoint hashes and independent reduction.
- `cached_verified`: original forecasts/forests, raw-derived seven-feature rows,
  splits, target/mask/envelope/row bindings and original prediction/action hashes.
- `not_run`: transfer, independent selection/calibration/confirmation, new neural
  training, full historical test suite, Stage5C and SMC. The advance screen failed.

Scope remains 12 previously exposed European development localities, obs8/pred12
raw stride12, image-local detector-silver. No metric/seconds, human-gold,
physical-safety, true3D, foundation or submission-ready claim. The deployment
floor remains unchanged. Historical SDD/external scores remain exploratory.
