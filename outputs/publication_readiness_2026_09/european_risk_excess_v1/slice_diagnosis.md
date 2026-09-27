# Post-Freeze Worst-View Diagnosis

Descriptive checks only; no threshold, model, feature or frozen action changes.
Locality-mean easy gates do not assert that every seed/producer view is safe.

| Candidate | Worst easy gain % | View | Views worse than -2% |
|---|---:|---|---:|
| dimensionless | -5.6144 | single2_seed29_controller1_dimensionless_heldeu-locality-110 | 5/72 |
| damped | -3.0685 | single0_seed17_controller2_damped_heldeu-locality-020 | 3/72 |

## Reference-Exact Cases

- single0_seed17_controller2_dimensionless_heldeu-locality-008: 1 harmed row-view, absolute image-local ADE harm 0.17435419; stationary-last-step 1, moving 0.

The diagnostic rule intentionally omits the old full policy stationary/utility/easy
guards. A stationary case would be denied by that already existing stationary guard,
but this check does not rerun or certify the complete policy. Do not discard a moving
failure or infer independent safety from an aggregate easy mean. Positive harm at zero
reference has no valid percentage denominator. No deployment change.
