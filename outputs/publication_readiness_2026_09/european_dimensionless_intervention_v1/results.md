# Matched Frozen-Predictor Intervention Results

Result provenance: 108 fresh Torch cost heads and causal decisions; cached_verified frozen forecasts.
Exploratory opened European source-development only. Obs8/pred12 at raw stride 12.
Detector-derived silver image-local trajectories, not human gold, metric, calibrated seconds,
independent confirmation, physical safety, true3D or foundation evidence. No deployment change.

## Full-Bank Pointwise and Unprotected Comparisons

Positive values mean lower error. Equal-locality percent gains; 3,000 paired locality-bootstrap draws.
Average seed/producer contexts inside each of 12 localities before resampling; overlapping windows
are not independent samples. CV is the fixed fallback; training-selected causal references are
reported separately, not silently called CV the strongest.

| Candidate | Policy | ADE vs CV | Easy ADE vs CV | Hard ADE vs CV | FDE vs CV |
|---|---|---:|---:|---:|---:|
| dimensionless | raw | 8.4616 [1.9118, 12.9795] | -11.1692 [-21.2576, -3.1438] | 13.3552 [8.8929, 16.5972] | 13.3983 [4.8669, 19.0176] |
| dimensionless | point | 0.3342 [0.2547, 0.4224] | 3.9961 [2.7207, 5.4116] | 0.1549 [0.0693, 0.2463] | 0.5088 [0.3646, 0.6807] |
| dimensionless | constant | -0.0000 [-0.0000, 0.0000] | -0.0000 [-0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [-0.0000, 0.0000] |
| damped | raw | 3.9755 [0.7820, 5.8095] | 1.5529 [-3.7281, 4.6920] | 5.5613 [4.0858, 6.5493] | 6.4267 [0.6467, 9.6446] |
| damped | point | 0.6994 [0.4878, 0.9297] | 3.4998 [2.7230, 4.1773] | 0.3933 [0.1914, 0.5996] | 1.1531 [0.8164, 1.5285] |
| damped | constant | 0.0000 [-0.0000, 0.0000] | 0.0001 [-0.0000, 0.0001] | 0.0000 [-0.0000, 0.0000] | 0.0000 [-0.0000, 0.0000] |

**Primary: pointwise neural vs equally protected damping ADE:** -0.3707 [-0.6136, -0.1252]%.

## Three Seeds

| Seed | Primary ADE gain, % |
|---|---:|
| 17 | -0.2609 [-0.5173, 0.0126] |
| 29 | -0.3455 [-0.5936, -0.0984] |
| 43 | -0.5057 [-0.7381, -0.2740] |

## Locality Safety

| Locality | Pointwise neural ADE vs CV, % | Easy gain, % |
|---|---:|---:|
| eu-locality-007 | 0.3482 | 1.1802 |
| eu-locality-008 | 0.4281 | 7.4405 |
| eu-locality-020 | 0.6207 | 1.0593 |
| eu-locality-048 | 0.2637 | 8.8290 |
| eu-locality-067 | 0.1381 | 4.9517 |
| eu-locality-074 | 0.3039 | 1.9955 |
| eu-locality-082 | 0.4871 | 2.3658 |
| eu-locality-110 | 0.5063 | 3.6734 |
| eu-locality-112 | 0.2164 | 3.2401 |
| eu-locality-119 | 0.1548 | 4.2223 |
| eu-locality-124 | 0.1533 | 2.5222 |
| eu-locality-126 | 0.3904 | 6.4727 |

## Joint Query Subset

96 hash-selected recording/frame queries per locality; all indexed query agents retained.
This is not full-bank joint evaluation. Exact-count diagnostics retain solver failures and
zero matches. A zero intervention rate or lower proximity proxy alone is not neural gain.

| Candidate | Dependent query views | Nonzero matched | Active nonadditive | Joint changes | Solver floors |
|---|---:|---:|---:|---:|---:|
| dimensionless | 6912 | 2422 | 41 | 14 | 53 |
| damped | 6912 | 3779 | 300 | 39 | 145 |

| Candidate | Joint vs independent ADE, all query subset | Joint vs unary ADE |
|---|---:|---:|
| dimensionless | -0.0013 [-0.0040, 0.0005] | 0.0000 [-0.0001, 0.0001] |
| damped | -0.0006 [-0.0019, 0.0002] | 0.0368 [0.0128, 0.0650] |

The preceding table retains solver-floor outcomes. It is not necessarily matched coverage.
Restricting to queries with verified equal nonfailed counts gives:

| Candidate | Matched joint vs independent ADE | Matched joint vs unary ADE |
|---|---:|---:|
| dimensionless | -0.0000 [-0.0006, 0.0006] | 0.0000 [-0.0001, 0.0001] |
| damped | -0.0007 [-0.0020, 0.0002] | 0.0004 [0.0001, 0.0008] |

Matched zero-action queries remain present. The matched-nonadditive population is
separate in summary_metrics.json; missing locality support is not silently dropped.

## Zero-Reference Costs

Reference-exact rows have undefined percentage gains. These are repeated producer/seed views,
not additional independent queries. The complete supported absolute-cost table follows.

| Candidate | Trial/controller | Site | Rows | Harmed rows | Mean selected ADE |
|---|---|---|---:|---:|---:|
| dimensionless | single0_seed17/1 | eu-locality-008 | 4 | 0 | 0.0000 |
| damped | single0_seed17/1 | eu-locality-008 | 4 | 0 | 0.0000 |
| dimensionless | single0_seed29/1 | eu-locality-008 | 4 | 0 | 0.0000 |
| damped | single0_seed29/1 | eu-locality-008 | 4 | 0 | 0.0000 |
| dimensionless | single0_seed43/1 | eu-locality-008 | 4 | 0 | 0.0000 |
| damped | single0_seed43/1 | eu-locality-008 | 4 | 0 | 0.0000 |
| dimensionless | single1_seed17/0 | eu-locality-008 | 4 | 0 | 0.0000 |
| damped | single1_seed17/0 | eu-locality-008 | 4 | 0 | 0.0000 |
| dimensionless | single1_seed29/0 | eu-locality-008 | 4 | 0 | 0.0000 |
| damped | single1_seed29/0 | eu-locality-008 | 4 | 0 | 0.0000 |
| dimensionless | single1_seed43/0 | eu-locality-008 | 4 | 0 | 0.0000 |
| damped | single1_seed43/0 | eu-locality-008 | 4 | 0 | 0.0000 |

## Gates

- primary_neural_over_protected_damping: false
- positive_neural_gain_over_CV: true
- every_locality_easy_preserved: true
- no_zero_reference_harm: true
- hard_nonnegative: true
- exploratory_useful_safe_screen: false
- independent_confirmation: false
- deployment_changed: false
- stage5c_executed: false
- smc_enabled: false

Fresh training: 108 heads, 216,000 updates; cumulative fit time 152.75s.
Training loss logs are optimization evidence, not a validation or generalization score.
Detailed query-level caches remain local; summary_metrics.json contains aggregate evidence.
