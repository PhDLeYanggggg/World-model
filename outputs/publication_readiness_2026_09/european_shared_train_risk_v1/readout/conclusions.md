# Shared TRAIN Risk: Fixed Recording-Transfer Evidence

## What Ran

Fresh_run: 108 cost heads, 216,000 optimizer updates, 489.45s fitting and 63.05s fixed-final inference.
The 72 local heads and 18 heads in each shared arm have identical per-source
query draws. Shared arms have identical initialization. Checkpoints and
all final receipts are hash-verified; held predictions replay exactly.
Source TRAIN packets and frozen forecasting models are cached_verified.
There is no newly trained trajectory model, JEPA representation or residual.

Each arm has 419,766 held-recording row occurrences, including 9,228 wholly unknown outcomes. These repeat across
contexts/seeds and are not independent samples. Preprocessing and fitting
use inner-fit recordings only; original outer development and independent
selection/calibration/confirmation roles were not opened by this experiment.

## Paired Comparisons

| Shared repair minus control | Full lower utility, pp (95% CI) | Same-count lower utility, pp (95% CI) | Signed-score MSE difference (95% CI) |
|---|---:|---:|---:|
| shared_global minus local | -2.094903 [-3.479909, -0.824900] | -0.519429 [-1.025350, -0.147155] | +7.845471 [+2.612407, +15.439524] |
| shared_source minus local | -2.604299 [-4.255594, -1.128204] | -0.381723 [-0.795017, -0.070482] | +3.438174 [+0.831990, +6.243236] |
| shared_source minus shared_global | -1.145644 [-1.672529, -0.637533] | -0.158733 [-0.340291, -0.006414] | -4.407297 [-12.244204, +0.217208] |

Utility is the pair-specific completion lower bound, divided by full
known floor-reference error mass. It is not FDE improvement. Shared unknown
actions cancel; unmatched unknown actions retain worst-case disagreement.
The 3,000-draw nominal bootstrap averages repeated views inside each of 12
source localities. Shared training and prior source exposure prevent an
independent significance or population safety claim. Head seeds are not
independent forecasting seeds.

## Safety and Support

| Arm | Selected occurrences | Known easy violations /72 | Completion-upper easy violations /72 | Undefined easy risk /72 | Full completion support /72 |
|---|---:|---:|---:|---:|---:|
| local | 57903 | 46 | 65 | 7 | 0 |
| shared_global | 34810 | 46 | 63 | 8 | 0 |
| shared_source | 27826 | 40 | 56 | 12 | 1 |

The unchanged 2% budget concerns positive selected harm/reference, not
net easy ADE. No missing label is imputed safe. Empty same-count risk remains
undefined and is reported separately from full-policy safety.

## Source Breakdown

| Source | Repeated views | Known easy violations: local / shared-global / shared-source | Completion support: local / shared-global / shared-source |
|---|---:|---:|---:|
| eu-locality-007 | 6 | 6 / 6 / 6 | 0 / 0 / 0 |
| eu-locality-008 | 6 | 0 / 2 / 1 | 0 / 0 / 0 |
| eu-locality-020 | 6 | 0 / 0 / 0 | 0 / 0 / 0 |
| eu-locality-048 | 6 | 5 / 5 / 6 | 0 / 0 / 0 |
| eu-locality-067 | 6 | 6 / 4 / 1 | 0 / 0 / 1 |
| eu-locality-074 | 6 | 0 / 6 / 5 | 0 / 0 / 0 |
| eu-locality-082 | 6 | 3 / 0 / 0 | 0 / 0 / 0 |
| eu-locality-110 | 6 | 6 / 6 / 6 | 0 / 0 / 0 |
| eu-locality-112 | 6 | 5 / 4 / 0 | 0 / 0 / 0 |
| eu-locality-119 | 6 | 6 / 6 / 6 | 0 / 0 / 0 |
| eu-locality-124 | 6 | 6 / 6 / 6 | 0 / 0 / 0 |
| eu-locality-126 | 6 | 3 / 1 / 3 | 0 / 0 / 0 |

These source-level slices include every registered view. They are diagnosis,
not permission to drop a failing source or choose a deployment threshold.

## Decision

- shared_global: eligible for later development design = False; failures: full_utility_not_supported, matched_utility_not_supported, MSE_missing_or_demonstrably_worse, full_policy_safety_support_failed.
- shared_source: eligible for later development design = False; failures: full_utility_not_supported, matched_utility_not_supported, MSE_missing_or_demonstrably_worse, full_policy_safety_support_failed, full_no_nondegradation_support_vs_shared_global, matched_no_nondegradation_support_vs_shared_global.

Neither this inner-TRAIN comparison nor routing eligibility authorizes
deployment or independent-test access. Negative controls remain visible.
Pooling changes the learned representation and prior as well as parameter
sharing; only shared-source versus shared-global isolates dual grouping.
Optimization loss contains a Lagrangian and can be negative; it is not a
negative ADE and is not used to declare success.

## Reproduction

```bash
.venv-pytorch/bin/python -m scripts.report_m3w_shared_train_risk --check
```

This rechecks all 108 weights, exact matched sampling, 72 readout receipts,
recording partitions, risk ratios, aggregate sums and bootstrap summary.
No new training or role access is required. Observation8/prediction12,
dataset-local coordinates and raw-frame stride12 remain unchanged. Not
true3D, not foundation, no metric/seconds claim; Stage5C and SMC disabled.
