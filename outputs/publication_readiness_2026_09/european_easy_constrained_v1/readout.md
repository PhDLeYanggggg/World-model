# Donor-Easy Constraint: Method and Actual Policy Readout

## Material Passport

Fresh local fits and fixed-policy re-evaluation on exposed TRAIN only. No independent test,
deployment upgrade, new neural forecaster, metric/seconds or physical-safety claim.
Full-policy 2% budget unchanged; empty count-matched risks are not estimable. Old failures retained.
Risk MSE is secondary, not a new paper-readiness gate.

## Donor Objective Conflict

```json
{
  "donor_conflicts": 16261,
  "reduced_alpha": 17221,
  "alpha_zero": 37569,
  "alpha_one": 2900,
  "alpha_sum": 10613.083869655968,
  "local_supported": 62902,
  "nested_supported": 59719
}
```

| Donor objective | All loss sum | Easy loss sum |
|---|---:|---:|
| before | 8642.8253 | 50.998271 |
| parent_after | 8096.656 | 51.250196 |
| after | 8285.5514 | 50.889516 |
| easy_only_after | 8540.7044 | 50.866247 |

These sums reuse overlapping donor sets; they are training objective diagnostics, not independent observations.
A preserved donor loss does not certify query or deployment risk.

## Actual Full Policy

| Arm | Selected | Unknown selected | Budget views | Budget record views | Undefined easy views | Known easy violations |
|---|---:|---:|---:|---:|---:|---:|
| origin | 62902 | 1567 | 1/72 | 16/444 | 7 | 25 |
| parent_blend | 48118 | 1039 | 5/72 | 34/444 | 7 | 21 |
| easy_only | 46321 | 943 | 8/72 | 40/444 | 7 | 18 |
| record_mean_easy | 48468 | 1010 | 7/72 | 29/444 | 7 | 23 |
| shuffled_easy | 54287 | 1255 | 6/72 | 24/444 | 7 | 24 |
| easy_constrained | 51693 | 1139 | 5/72 | 29/444 | 7 | 22 |

Budget support checks all/easy positive harm/reference and signed easy degradation separately from utility.
All recorded scenes stay; undefined full-policy risk does not pass. No population guarantee follows.

## Primary Policy Utility

| Comparison | Equal-view completion-lower change(pp) |
|---|---:|
| origin_full | -2.8797677 |
| origin_matched | -0.69211001 |
| parent_blend_full | -0.46693248 |
| parent_blend_matched | 0.097504998 |

Empty matched views: {"origin": 0, "parent_blend": 2}.
Paired unknown outcomes cancel on shared actions; exchanged unknown rows retain envelope bounds.
Empty matched utility contributes zero where its full reference denominator exists; not positive evidence.
These are fitting partial-label utility diagnostics, not independent test ADE/FDE improvements.

## Observed Versus Unresolved Risk

Counts overlap. Unknown-only failures have a defined known-label ratio within budget,
but selected missing labels make the completion upper bound exceed budget.

```json
{
  "origin": {
    "known_all_over_budget": 35,
    "known_easy_over_budget": 25,
    "completion_all_over_budget": 66,
    "completion_easy_over_budget": 64,
    "easy_degradation_upper_over_budget": 68,
    "unknown_only_all_failure": 31,
    "unknown_only_easy_failure": 39
  },
  "parent_blend": {
    "known_all_over_budget": 36,
    "known_easy_over_budget": 21,
    "completion_all_over_budget": 63,
    "completion_easy_over_budget": 59,
    "easy_degradation_upper_over_budget": 61,
    "unknown_only_all_failure": 27,
    "unknown_only_easy_failure": 38
  },
  "easy_only": {
    "known_all_over_budget": 35,
    "known_easy_over_budget": 18,
    "completion_all_over_budget": 62,
    "completion_easy_over_budget": 56,
    "easy_degradation_upper_over_budget": 60,
    "unknown_only_all_failure": 27,
    "unknown_only_easy_failure": 38
  },
  "record_mean_easy": {
    "known_all_over_budget": 34,
    "known_easy_over_budget": 23,
    "completion_all_over_budget": 61,
    "completion_easy_over_budget": 57,
    "easy_degradation_upper_over_budget": 57,
    "unknown_only_all_failure": 27,
    "unknown_only_easy_failure": 34
  },
  "shuffled_easy": {
    "known_all_over_budget": 34,
    "known_easy_over_budget": 24,
    "completion_all_over_budget": 62,
    "completion_easy_over_budget": 59,
    "easy_degradation_upper_over_budget": 63,
    "unknown_only_all_failure": 28,
    "unknown_only_easy_failure": 35
  },
  "easy_constrained": {
    "known_all_over_budget": 33,
    "known_easy_over_budget": 22,
    "completion_all_over_budget": 63,
    "completion_easy_over_budget": 59,
    "easy_degradation_upper_over_budget": 63,
    "unknown_only_all_failure": 30,
    "unknown_only_easy_failure": 37
  }
}
```

## Paired Partial-Label Intervals

| Contrast | Lower (pp) | Known component (pp) | Upper (pp) |
|---|---:|---:|---:|
| origin_full | -2.8797677 | -2.2253108 | -1.5708538 |
| origin_matched | -0.69211001 | -0.57316669 | -0.45422338 |
| parent_blend_full | -0.46693248 | 0.8582162 | 2.1833649 |
| parent_blend_matched | 0.097504998 | 0.13154429 | 0.16558359 |

These are deterministic completion bounds, not sampling confidence intervals.
A negative lower bound alone does not prove the true policy effect is negative.

## Secondary Risk Estimation

| Arm | All signed-risk MSE | Easy signed-risk MSE |
|---|---:|---:|
| origin | 0.079458745 | 0.00030937342 |
| parent_blend | 0.080533858 | 0.00032696178 |
| easy_only | 0.082626338 | 0.00031353751 |
| record_mean_easy | 0.078130705 | 0.00031269889 |
| shuffled_easy | 0.081134981 | 0.00032383732 |
| easy_constrained | 0.079382933 | 0.00031725353 |

## Complete Source/Context/Head-Seed Breakdown

| Slice | Original easy MSE | Primary easy MSE | Primary full utility vs original(pp) |
|---|---:|---:|---:|
| source:eu-locality-007 | 0.0014372615 | 0.001437129 | -3.4790691 |
| source:eu-locality-008 | 0.00038952993 | 0.00038967985 | -0.84522454 |
| source:eu-locality-020 | 9.2869514e-06 | 4.0810313e-05 | -1.203253 |
| source:eu-locality-048 | 0.00016394108 | 0.00016411432 | -4.0081173 |
| source:eu-locality-067 | 1.166204e-05 | 1.1985749e-05 | -1.4982734 |
| source:eu-locality-074 | 0.0016662926 | 0.0016667536 | -1.4112198 |
| source:eu-locality-082 | 3.9897876e-08 | 5.822949e-05 | -2.2801632 |
| source:eu-locality-110 | 7.632445e-06 | 7.6478112e-06 | -4.6040526 |
| source:eu-locality-112 | 2.4667987e-07 | 6.4996533e-07 | -4.6600205 |
| source:eu-locality-119 | 9.7514613e-06 | 9.7517265e-06 | -0.60053586 |
| source:eu-locality-124 | 2.2373174e-07 | 2.9209855e-07 | -2.6245442 |
| source:eu-locality-126 | 1.6612689e-05 | 1.9998491e-05 | -7.3427388 |
| context:single0_seed43_controller1_dimensionless | 0.00017506015 | 0.00019105936 | -1.8746436 |
| context:single0_seed43_controller2_dimensionless | 0.00012848773 | 0.00012843164 | -2.7738359 |
| context:single1_seed43_controller0_dimensionless | 0.00011214985 | 0.00011223576 | -1.9027906 |
| context:single1_seed43_controller2_dimensionless | 0.00019603993 | 0.00019605898 | -4.0105549 |
| context:single2_seed43_controller0_dimensionless | 0.00059894684 | 0.00059897865 | -2.880265 |
| context:single2_seed43_controller1_dimensionless | 0.000645556 | 0.00067675679 | -3.8365162 |
| seed:17 | 0.00031307429 | 0.00032796542 | -3.5761461 |
| seed:29 | 0.00033723126 | 0.00034593927 | -2.7514208 |
| seed:43 | 0.0002778147 | 0.0002778559 | -2.3117362 |

No source deletion, best-arm substitution or query/test threshold tuning.
62902 repeated query occurrences, not independent samples; head seeds are not new forecaster seeds.
obs8/pred12 rawstride12, pixel/local detector-silver. Stage5C and SMC remain off.
