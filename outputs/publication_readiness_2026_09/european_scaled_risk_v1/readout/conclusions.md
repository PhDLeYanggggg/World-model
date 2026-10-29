# Fixed TRAIN Constraint Scaling: Complete Results

## Verdict

The candidate **fails the registered development advancement rule**. No deployment change follows. This is exposed
development evidence, not independent confirmation or submission readiness.

## What Ran

144 fresh fits: 72 unit controls and 72 budget-scaled heads, each 2,000 updates.
All 72 unit controls exactly reproduce the prior task-risk model, optimizer,
sampler, duals and traces. The quadratic/utility heads are 144 cached-verified
controls, not newly trained. Four forest controls remain visible. Forecast
producers, features, initialization, draws, loss components, temperature, hard
rule and risk definition stay fixed; only positive TRAIN constraint units change
in the primal and dual updates. Zero-support scaling is not imputed safe.

All 72 views over 12 localities complete, with 596,988 repeated row occurrences
and 14,076 unknown-outcome occurrences. Observation8/prediction12 and
dataset-local coordinates are unchanged; these are not t+50 or metric results.

## Full and Same-Count Comparisons

Candidate minus control. Positive paired lower utility is favorable; negative
MSE is favorable. Units are normalized utility percentage points, not ADE/FDE.
All eight full/matched contrasts against unit, quadratic, utility and original
forest are primary. A weak control cannot rescue a failed strong control.

| Control | Signed-score MSE delta | Full paired lower utility delta (pp) | Same-count delta (pp) |
|---|---:|---:|---:|
| original | +2.628736 [+0.924916, +5.341594] | +2.258131 [+0.718677, +3.848069] | +0.778894 [+0.252400, +1.448081] |
| additive | +2.660883 [+0.942921, +5.388607] | +1.893155 [+0.368093, +3.461074] | +0.738883 [+0.149989, +1.420627] |
| poisson | +2.492575 [+0.779026, +5.206085] | +2.212046 [+0.661810, +3.791813] | +0.798940 [+0.218784, +1.492792] |
| cost | +2.524583 [+0.809799, +5.247252] | +2.252237 [+0.714346, +3.839045] | +0.785237 [+0.239173, +1.473522] |
| quadratic | +2.314067 [+0.788144, +4.806557] | -0.184760 [-1.170105, +0.621735] | +0.185634 [-0.134510, +0.524161] |
| task_utility | +2.339304 [+0.781677, +4.860882] | -2.940035 [-4.350033, -1.682244] | -0.699983 [-1.134574, -0.257939] |
| unit_control | +2.370133 [+0.797264, +4.891259] | -1.890975 [-3.157684, -0.782172] | -0.484529 [-0.827914, -0.142285] |

Intervals use 3,000 paired-locality draws after averaging repeated views within
locality. They are nominal, not simultaneous or independent confirmation.

## Risk and Support

The 2% limit concerns positive selected easy harm/reference error, not net easy
ADE degradation. Unknown outcomes remain in completion bounds. Empty or
unsupported cohorts do not pass. Original-unit constraints are reported below.

| Policy | Selected occurrences | Known easy violations | Upper violations | Finite support /72 |
|---|---:|---:|---:|---:|
| original | 95455 | 4 | 7 | 33 |
| additive | 112456 | 20 | 42 | 19 |
| poisson | 111031 | 2 | 11 | 37 |
| cost | 96720 | 4 | 7 | 36 |
| quadratic | 107569 | 60 | 72 | 0 |
| task_utility | 160275 | 62 | 72 | 0 |
| unit_control | 109401 | 58 | 71 | 1 |
| budget_scaled | 112749 | 58 | 67 | 0 |
| original_matched_budget_scaled | 68571 | 6 | 9 | 22 |
| budget_scaled_matched_original | 68571 | 23 | 30 | 1 |
| additive_matched_budget_scaled | 73622 | 18 | 36 | 17 |
| budget_scaled_matched_additive | 73622 | 34 | 45 | 2 |
| poisson_matched_budget_scaled | 72932 | 1 | 9 | 28 |
| budget_scaled_matched_poisson | 72932 | 30 | 38 | 2 |
| cost_matched_budget_scaled | 69342 | 5 | 7 | 25 |
| budget_scaled_matched_cost | 69342 | 26 | 34 | 2 |
| quadratic_matched_budget_scaled | 62139 | 55 | 66 | 1 |
| budget_scaled_matched_quadratic | 62139 | 56 | 66 | 1 |
| task_utility_matched_budget_scaled | 78477 | 59 | 66 | 1 |
| budget_scaled_matched_task_utility | 78477 | 56 | 66 | 1 |
| unit_control_matched_budget_scaled | 69334 | 59 | 65 | 2 |
| budget_scaled_matched_unit_control | 69334 | 56 | 66 | 1 |

## Mechanism Checks

| Arm | TRAIN all excess | TRAIN easy excess | Dual-cap views | Empty views | Mean soft/hard disagreement |
|---|---:|---:|---:|---:|---:|
| quadratic | 52 | 63 | 0 | 0 | 0.239213 |
| task_utility | 58 | 64 | 0 | 0 | 0.347696 |
| unit_control | 38 | 61 | 0 | 0 | 0.283336 |
| budget_scaled | 13 | 12 | 1 | 4 | 0.188633 |

TRAIN counts refer to fixed 128-query monitors, not the full TRAIN distribution.
The soft-at-least-0.5 comparison is diagnostic, not calibrated confidence or a
new threshold. A zero dual at the final step is not evidence that its constraint
always held. Total losses differ across arms and must not be ranked directly.
Scaling changes conditioning and effective dual capacity, not the feasible set.
Fewer violations with no intervention do not establish a useful world model.
The unit-control comparison isolates the scaling intervention; it does not prove
that optimization scale is the sole cause of any remaining generalization gap.

## Advancement Failures

- unsupported_unit_control_full_paired_lower_percent
- unsupported_unit_control_matched_paired_lower_percent
- unsupported_quadratic_full_paired_lower_percent
- unsupported_quadratic_matched_paired_lower_percent
- unsupported_task_utility_full_paired_lower_percent
- unsupported_task_utility_matched_paired_lower_percent
- incomplete_or_unsafe_completion_budget_scaled
- selected_easy_risk_violated_budget_scaled
- incomplete_or_unsafe_completion_budget_scaled_matched_unit_control
- selected_easy_risk_violated_budget_scaled_matched_unit_control
- incomplete_or_unsafe_completion_budget_scaled_matched_quadratic
- selected_easy_risk_violated_budget_scaled_matched_quadratic
- incomplete_or_unsafe_completion_budget_scaled_matched_task_utility
- selected_easy_risk_violated_budget_scaled_matched_task_utility
- incomplete_or_unsafe_completion_budget_scaled_matched_original
- selected_easy_risk_violated_budget_scaled_matched_original

## Verification and Reproduction

All model, input, split and role bindings verify. Repeated inference is exact,
all forest controls replay, and all three prior neural prediction hashes match.
Independent scalar checks total 13,276,438; readout
took 223.59s. The summary replays exactly from frozen aggregates.
No checkpoint selection, threshold sweep or independent-role access occurred.
Training is native arm64 Torch, four compute threads and zero loader workers.
Models/checkpoints and raw data stay outside Git; light receipts and metrics remain.

```bash
.venv-pytorch/bin/python -m scripts.report_m3w_scaled_risk --check
```

This checks existing results, not new training. The full method remains a 2.5D
world-state scaffold; this cost-head repair is not new trajectory dynamics,
foundation capability, true3D, physical safety or seconds-level evidence.
Silver labels are not human gold. Stage5C and SMC remain disabled.
