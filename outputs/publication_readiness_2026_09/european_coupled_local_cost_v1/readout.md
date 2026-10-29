# Coupled Local Cost Retrieval: Complete Fitting Results

## Material Passport

Fresh_run: exact other-recording retrieval, 288 comparison groups and separate
full retrieval/readout replay. Cached_verified: original source packets, forecast
producers and the paired cost-head controls. Not_run: new neural optimization,
trajectory training, independent validation/confirmation or deployment.

First PID 33535, exit0, 260.627765s, peak RSS 10,343,464,960 B.
Replay PID 34342, 258.380133s, peak RSS 10,829,643,776 B.
Native arm64 CPU4, workers0, checkpointed neighbors and private heartbeat.
The new estimator is deterministic: repeated comparator seeds are not three
independent retrieval models or independently trained forecasting pipelines.

## Fixed Comparison

All queries and original labels remain. Each query excludes its entire recording
from fitted donors. The fixed 398-field distance descriptor averages standardized
physical-action context. Shared-reference and convex harm decoding preserve
moment ordering, but do not certify selected-risk calibration. Normalizers and
upstream producers saw fitting data, so this is not whole-pipeline OOF.

## Accuracy And Full Risk

| Mode | Bank | Known ADE gain over floor (%), fitting CI | Required risk support | Undefined easy | Known all/easy violations | Selected (unknown) |
|---|---|---|---:|---:|---:|---:|
| local_known | physical_only | +1.255661 [+0.583036, +1.969732] | 12/444 | 6 | 240/255 | 396741 (5223) |
| local_known | physical_neural | +1.334797 [+0.627570, +2.106079] | 6/444 | 6 | 276/321 | 407937 (5406) |
| local_bounds | physical_only | +0.777027 [+0.211757, +1.336716] | 27/444 | 18 | 219/216 | 295344 (3357) |
| local_bounds | physical_neural | +0.830158 [+0.234853, +1.429495] | 30/444 | 15 | 237/273 | 303363 (3474) |
| global_bounds | physical_only | +0.004444 [+0.001710, +0.007419] | 312/444 | 123 | 3/3 | 34098 (414) |
| global_bounds | physical_neural | +0.004133 [+0.001722, +0.006856] | 291/444 | 123 | 6/9 | 35592 (432) |
| shuffled_bounds | physical_only | +0.780320 [+0.321319, +1.287543] | 9/444 | 12 | 348/318 | 314733 (3927) |
| shuffled_bounds | physical_neural | +0.812390 [+0.320192, +1.342853] | 0/444 | 12 | 369/363 | 323811 (4053) |

No empty selected-risk estimate passes by default. All required recordings,
unknown outcomes and undefined ratios remain. Count matching is utility only.

## Mixed Policy Minus Frozen Known-Only Cost Head

| Mode | Full known change (pp), fitting CI | Full completion-lower (pp), fitting CI | Matched completion-lower (pp), fitting CI |
|---|---|---|---|
| local_known | +0.753998 [-0.137626, +1.614157] | +0.089779 [-0.826965, +0.842210] | -0.252839 [-0.742048, +0.091894] |
| local_bounds | +0.249359 [-0.539966, +0.942678] | +0.006377 [-0.721776, +0.635341] | -0.113943 [-0.443232, +0.136619] |
| global_bounds | -0.576666 [-0.974525, -0.264750] | -0.633832 [-1.033325, -0.307427] | -0.055232 [-0.136572, -0.005696] |
| shuffled_bounds | +0.231591 [-0.285944, +0.807879] | -0.092652 [-0.616300, +0.436296] | -0.156290 [-0.309425, -0.031819] |

## Mixed Minus Physical Bank

| Mode | Full known increment (pp), fitting CI | Full completion-lower (pp), fitting CI |
|---|---|---|
| local_known | +0.079137 [+0.033069, +0.133951] | +0.048878 [+0.007155, +0.096342] |
| local_bounds | +0.053131 [+0.013893, +0.098689] | +0.040875 [+0.001154, +0.086588] |
| global_bounds | -0.000310 [-0.000756, +0.000081] | -0.000355 [-0.000831, +0.000062] |
| shuffled_bounds | +0.032070 [-0.001644, +0.067851] | +0.027379 [-0.004451, +0.059593] |

## Primary Local-Bounds Minus Global-Bounds

- Full known: +0.826025 [+0.231961, +1.424076] pp.
- Full completion-lower: +0.630100 [+0.089415, +1.187286] pp.
- Matched completion-lower: +0.099966 [-0.010578, +0.245421] pp.

## Known Hard/Easy Slices

Supplementary pooled errors over repeated views, not source-macro accuracy,
independent sample counts or substitutes for full recording risk.

| Mode | Subset | Known row views | Empty source views | Mixed ADE gain (%) | Mixed FDE gain (%) |
|---|---|---:|---:|---:|---:|
| local_known | hard | 187137 | 0 | +1.703907 | +2.408743 |
| local_known | easy | 214257 | 0 | +12.869863 | +20.305627 |
| local_bounds | hard | 187137 | 0 | +1.157481 | +1.651505 |
| local_bounds | easy | 214257 | 0 | +9.464783 | +15.039496 |
| global_bounds | hard | 187137 | 0 | +0.000052 | +0.000042 |
| global_bounds | easy | 214257 | 0 | +0.215439 | +0.418720 |
| shuffled_bounds | hard | 187137 | 0 | +0.879267 | +1.333054 |
| shuffled_bounds | easy | 214257 | 0 | +6.963772 | +11.708131 |

## Proxy Fit

Normalized component MSE uses only real candidate actions. Unknown targets
are interval endpoints, not actual future errors or a calibration certificate.

| Mode | Mixed-bank support | Real action occurrences | Undefined groups | B / H / R / ER / EH MSE |
|---|---|---:|---:|---|
| local_known | all | 5369850 | 0 | 0.29361, 1.69333, 2.14208, 0.00316808, 1.5806 |
| local_known | known | 5256810 | 0 | 0.293444, 0.553181, 2.15448, 0.00320686, 0.0316951 |
| local_known | unknown | 113040 | 0 | 0.301307, 54.7147, 1.56512, 0.00136485, 73.6109 |
| local_bounds | all | 5369850 | 0 | 0.292427, 1.53101, 2.14946, 0.0031553, 1.37214 |
| local_bounds | known | 5256810 | 0 | 0.292857, 0.689996, 2.16559, 0.00319521, 0.216394 |
| local_bounds | unknown | 113040 | 0 | 0.272432, 40.6413, 1.39931, 0.00129939, 55.119 |
| global_bounds | all | 5369850 | 0 | 0.317034, 1.63153, 2.23782, 0.00308509, 1.48247 |
| global_bounds | known | 5256810 | 0 | 0.317389, 0.52686, 2.2641, 0.00312004, 0.0669131 |
| global_bounds | unknown | 113040 | 0 | 0.300553, 53.0033, 1.01594, 0.00145956, 67.3117 |
| shuffled_bounds | all | 5369850 | 0 | 0.32028, 1.6472, 2.51116, 0.00333659, 1.3685 |
| shuffled_bounds | known | 5256810 | 0 | 0.321491, 0.755534, 2.53742, 0.00338349, 0.221904 |
| shuffled_bounds | unknown | 113040 | 0 | 0.263978, 43.113, 1.2898, 0.00115586, 54.69 |

## Gates

- full288groups: true
- complete_mixed_record_risk: false
- full_prior_change_lower_CI_positive: false
- matched_prior_change_lower_CI_positive: false
- neural_increment_lower_CI_positive: true
- local_over_global_lower_CI_positive: true
- fitting_advancement_gate: false

The primary and all secondary controls remain visible; no better-looking
secondary is promoted after readout. All intervals are 3,000 fitting-locality
bootstrap draws after dependent context/comparator-seed averaging. They are
not independent generalization, simultaneous certification or correction for
repeated research selection. Undefined required FDE remains not_estimable.

SOURCE is oblique detector-track midpoint pixels, obs8/pred12/stride12 raw
frames. No metric/seconds, human-gold, true-3D, foundation, physical-safety or
new world-dynamics claim. Deployment, independent roles, Stage5C and SMC are
unchanged. See failure_analysis.md and reproduction_zh.md for interpretation
and verified command scope.
