# Matched Zero-Target Gradient Intervention

## Material Passport

Twelve new cold-start fits, 120,000 updates, three seeds; twelve controls cached_verified.
Same source folds, inputs, preprocessing, scale, initial seeds, sampler and terminal budget.
All held queries retained. No future label is an inference input. No learned deployment gate.

| Arm | Equal-site all gain | Conditional site 95% CI | Oracle gain (diagnostic) | Easy pixel harm |
| --- | ---: | --- | ---: | ---: |
| unconditional_control | -5.015980% | [-8.396553368897639, -2.4877265310346948] | 0.467646% | 0.091902 |
| motion_loss | -98.719201% | [-128.49922805885276, -66.02161610972706] | 3.759681% | 1.428799 |

Matched contrasts:

{
  "motion_minus_control_all": {
    "point_pp": -93.70322094708183,
    "conditional_site_ci95": [
      -120.10267468995515,
      -62.459141852792435
    ]
  },
  "motion_minus_control_oracle": {
    "point_pp": 3.2920354719936546,
    "conditional_site_ci95": [
      3.0464487759854513,
      3.6415650767691123
    ]
  }
}

Oracle values use future labels and cannot be deployed. Zero-target easy percentage is undefined.
Shared training folds and explored sites prevent independent confirmation. This loss weighting
control is not a novel architecture or evidence of useful causal selection by itself.
Pixels/past-normalized raw frames only. No metric/seconds/true-3D/foundation claim. Stage5C/SMC off.
