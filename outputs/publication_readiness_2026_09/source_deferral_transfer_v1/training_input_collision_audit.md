# Exact Observed-Input Collision Audit

## Material Passport

Post-hoc, training-only analysis after the fixed source readout. No model fitting,
held forecast, label-based input construction or policy change. It concerns the
current mask-only predictor, not a hypothetical richer multimodal observation.

| Quantity | Value |
| --- | ---: |
| training_rows | 15430 |
| unique_observed_inputs | 15309 |
| duplicate_groups | 32 |
| rows_in_duplicate_groups | 153 |
| conflicting_groups | 1 |
| rows_in_conflicting_groups | 38 |
| zero_optimal_conflicting_groups | 0 |
| rows_in_zero_optimal_conflicting_groups | 0 |
| baseline_cost_fraction_in_zero_optimal_conflicts | 0.0 |
| largest_duplicate_group | 38 |

## Exact Scope and Argument

Keys contain the actual geometry, every coverage-mask element, radius, rotation
and support flag. Floating signed zero is canonicalized; no approximate
clustering, PCA or learned metric is used. RGB is excluded only because the
current mask-only model explicitly replaces it with zero. The keys include
the restoration frame so identical hidden inputs with different restored
outputs are not incorrectly merged. Future labels are read after grouping.

For a group with identical effective inputs and uniform weights, any
deterministic current-input predictor must return one common trajectory.
If at least half the full target paths are exactly zero, stationary CV is
an empirical minimizer of mean Euclidean ADE on that group: at each step,
triangle inequality gives total cost at q at least baseline cost plus
(number_zero - number_nonzero)*norm(q). Only conflicting duplicate groups
are counted for this sufficient condition. Zero may be a nonunique minimizer.

This is a finite-training-cohort statement, not a Bayes-risk or real-world
unpredictability theorem. It is sensitive to observed schema and precision;
exact duplicates cannot describe near-duplicate ambiguity. Small collision
coverage would reject exact aliasing as the main explanation, not prove
that current features contain enough learnable signal. New RGB/video inputs
could distinguish rows that this mask-only schema cannot.

Offline supplied annotations, pixel/past-normalized raw-frame task only.
No metric/seconds/true-3D/foundation claim. Stage5C and SMC remain off.
