# Direct Neural Intervention: Fixed-Final Readout

## What Ran

54 new neural policy fits,108,000 optimizer updates; 198.571s training,
102.727s readout; peak training RSS 3,119,202,304 bytes.
Fresh_run: policy fitting and fixed-final inference. Cached_verified: original
forecaster/risk artifacts, source data and separate receipt arithmetic.
No new trajectory forecaster, JEPA, residual or independent evaluation.
All models frozen before inner-held scoring. Three head seeds are not three
independent forecasting seeds. Prior exposure makes inner-held exploratory.

## Full Policy Safety

Original 2% all/easy positive-harm/reference and easy-preservation requirements.
Risk is error-related, not physical safety. Undefined risks do not pass.

| Cohort | Policy | Rows | Selected | Unknown selected | Budget views | Budget recording views | Known all/easy violations | Undefined easy |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| fit | floor | 897060 | 0 | 0 | 0/72 | 0/444 | 0/0 | 72 |
| fit | origin | 897060 | 62902 | 1567 | 1/72 | 16/444 | 35/25 | 7 |
| fit | soft_source | 897060 | 40187 | 108 | 46/72 | 164/444 | 19/0 | 7 |
| fit | soft_record | 897060 | 41223 | 125 | 48/72 | 171/444 | 15/1 | 10 |
| fit | hard_record | 897060 | 46508 | 226 | 35/72 | 125/444 | 28/5 | 9 |
| inner_held | floor | 419766 | 0 | 0 | 0/72 | 0/186 | 0/0 | 72 |
| inner_held | origin | 419766 | 32621 | 847 | 0/72 | 3/186 | 61/44 | 14 |
| inner_held | soft_source | 419766 | 17111 | 288 | 5/72 | 22/186 | 39/37 | 17 |
| inner_held | soft_record | 419766 | 17510 | 309 | 4/72 | 24/186 | 40/39 | 17 |
| inner_held | hard_record | 419766 | 19915 | 326 | 4/72 | 17/186 | 43/50 | 11 |

Row and recording occurrences repeat across contexts/seeds; not independent sample counts.
The floor makes zero interventions. Its undefined selected risk is not zero or a safety certificate.

## Paired Utility

| Cohort | Contrast | Full lower/upper (pp) | Equal-count lower/upper (pp) | Empty matched views |
|---|---|---:|---:|---:|
| fit | soft_source_minus_origin | -10.557322 / -2.068275 | -2.060830 / -1.085135 | 6 |
| fit | soft_record_minus_origin | -10.484009 / -1.994833 | -2.008661 / -0.995423 | 6 |
| fit | hard_record_minus_origin | -10.243319 / -1.753750 | -2.182481 / -1.095508 | 7 |
| fit | soft_record_minus_soft_source | +0.073235 / +0.073519 | +0.024910 / +0.024985 | 5 |
| fit | hard_record_minus_soft_record | +0.240403 / +0.241370 | +0.021645 / +0.021893 | 4 |
| inner_held | soft_source_minus_origin | -5.490250 / -0.452656 | -1.062643 / -0.693997 | 13 |
| inner_held | soft_record_minus_origin | -5.450130 / -0.473388 | -1.125104 / -0.577616 | 19 |
| inner_held | hard_record_minus_origin | -5.342713 / -0.326202 | -1.132721 / -0.625042 | 15 |
| inner_held | soft_record_minus_soft_source | -0.101213 / +0.120601 | -0.000948 / +0.032262 | 8 |
| inner_held | hard_record_minus_soft_record | -0.049812 / +0.304416 | +0.012406 / +0.030458 | 12 |

Intervals are deterministic partial-label completion bounds, not confidence intervals.
Each view uses the full known floor-reference error denominator, then equal-view averaging.
They are not relative FDE improvements. Matched risks are descriptive, never safety gates.
Unknown shared actions cancel in paired utility; exchanged unknowns retain envelope uncertainty.

## Training Diagnostics

```json
{
  "soft_source": {
    "fits": 18,
    "parameters_each": [
      12385
    ],
    "mean_first_loss": -0.001387003990304139,
    "mean_final_loss": -0.17808598848498353,
    "mean_final_gain_lower": 0.012533955647818907,
    "mean_final_action_fraction": 0.08023107988022578,
    "mean_soft_hard_gap": 0.005147601769067478,
    "final_duals_at_cap": 6,
    "final_dual_count": 144,
    "mean_max_final_dual": 87.87640889485677,
    "total_row_draws": 56832000
  },
  "soft_record": {
    "fits": 18,
    "parameters_each": [
      12385
    ],
    "mean_first_loss": -0.001387003990304139,
    "mean_final_loss": -0.17205882634435612,
    "mean_final_gain_lower": 0.01425433269668651,
    "mean_final_action_fraction": 0.07866273833173586,
    "mean_soft_hard_gap": 0.004393422178218821,
    "final_duals_at_cap": 68,
    "final_dual_count": 888,
    "mean_max_final_dual": 99.80726114908855,
    "total_row_draws": 56832000
  },
  "hard_record": {
    "fits": 18,
    "parameters_each": [
      12385
    ],
    "mean_first_loss": -0.036880850378010005,
    "mean_final_loss": -0.17200291178111607,
    "mean_final_gain_lower": 0.01603268577653378,
    "mean_final_action_fraction": 0.08720313300566179,
    "mean_soft_hard_gap": 0.005511940427892468,
    "final_duals_at_cap": 30,
    "final_dual_count": 888,
    "mean_max_final_dual": 94.91478686862521,
    "total_row_draws": 56832000
  }
}
```

Dual-weighted objectives vary during training, so first/final loss is not a fixed validation loss.
Final minibatch constraints do not prove whole-recording or held-recording safety.

## Complete Source and Head-Seed Slices

| Cohort | Slice | Primary full utility vs original (pp) | Primary matched utility vs original (pp) | Primary budget views |
|---|---|---:|---:|---:|
| fit | source:eu-locality-007 | -13.506292 / -4.793547 | -0.288092 / -0.161386 | 1/6 |
| fit | source:eu-locality-008 | -2.237610 / -0.518819 | -1.250104 / -0.360044 | 0/6 |
| fit | source:eu-locality-020 | -5.063448 / +1.070897 | +0.157247 / +0.157247 | 6/6 |
| fit | source:eu-locality-048 | -12.284468 / -4.840821 | -9.310817 / -3.412154 | 6/6 |
| fit | source:eu-locality-067 | -4.433733 / -2.629600 | -0.616363 / -0.255872 | 3/6 |
| fit | source:eu-locality-074 | -3.420479 / -1.686435 | -0.283280 / -0.059979 | 0/6 |
| fit | source:eu-locality-082 | -26.640243 / +19.094140 | +0.000000 / +0.000000 | 0/6 |
| fit | source:eu-locality-110 | -8.730932 / -7.139028 | -3.597392 / -2.881611 | 5/6 |
| fit | source:eu-locality-112 | -10.135092 / +2.808773 | -0.597795 / -0.538333 | 6/6 |
| fit | source:eu-locality-119 | -8.769881 / -4.613508 | -3.720170 / -2.106150 | 0/6 |
| fit | source:eu-locality-124 | -13.899990 / -7.995433 | -1.472586 / +0.213423 | 3/6 |
| fit | source:eu-locality-126 | -13.797662 / -9.801622 | -5.210415 / -3.741243 | 5/6 |
| fit | context:single0_seed43_controller1_dimensionless | -7.195852 / -2.160335 | -3.591016 / -1.311379 | 9/12 |
| fit | context:single0_seed43_controller2_dimensionless | -12.111607 / -6.198903 | -4.293836 / -2.222578 | 4/12 |
| fit | context:single1_seed43_controller0_dimensionless | -4.813778 / -1.298836 | -1.919003 / -0.624032 | 6/12 |
| fit | context:single1_seed43_controller2_dimensionless | -8.586235 / -2.366661 | -2.896682 / -2.262808 | 6/12 |
| fit | context:single2_seed43_controller0_dimensionless | -10.341941 / -6.071855 | -0.245284 / -0.245284 | 5/12 |
| fit | context:single2_seed43_controller1_dimensionless | -18.410503 / +7.574089 | -0.149063 / +0.093030 | 5/12 |
| fit | seed:17 | -10.458258 / -1.312304 | -1.933981 / -0.950084 | 10/24 |
| fit | seed:29 | -11.075421 / -1.804911 | -2.366395 / -1.137675 | 13/24 |
| fit | seed:43 | -9.196279 / -2.144035 | -2.247066 / -1.198766 | 12/24 |
| inner_held | source:eu-locality-007 | -0.635702 / -0.324202 | -0.110858 / -0.039777 | 0/6 |
| inner_held | source:eu-locality-008 | -2.154457 / -0.079092 | -0.696237 / +0.045951 | 0/6 |
| inner_held | source:eu-locality-020 | -18.645092 / +14.682681 | +0.000000 / +0.000000 | 0/6 |
| inner_held | source:eu-locality-048 | -4.936193 / -1.757870 | -2.377656 / -1.729299 | 1/6 |
| inner_held | source:eu-locality-067 | -5.339067 / -0.879753 | -0.921334 / -0.887907 | 0/6 |
| inner_held | source:eu-locality-074 | -3.325543 / -2.068162 | -0.283461 / -0.141473 | 0/6 |
| inner_held | source:eu-locality-082 | -0.220160 / +0.995065 | +0.000000 / +0.000000 | 2/6 |
| inner_held | source:eu-locality-110 | -5.159041 / -3.954801 | -3.203762 / -2.543401 | 0/6 |
| inner_held | source:eu-locality-112 | -2.706276 / +0.167413 | -0.718199 / +0.573947 | 0/6 |
| inner_held | source:eu-locality-119 | -4.686301 / -2.037119 | -3.419467 / -1.182315 | 0/6 |
| inner_held | source:eu-locality-124 | -6.529897 / +0.165811 | -0.806225 / -0.540780 | 0/6 |
| inner_held | source:eu-locality-126 | -9.774826 / -8.824395 | -1.055447 / -1.055447 | 1/6 |
| inner_held | context:single0_seed43_controller1_dimensionless | -8.424936 / +2.681205 | -1.770845 / -1.199736 | 0/12 |
| inner_held | context:single0_seed43_controller2_dimensionless | -6.339600 / -2.307535 | -3.328770 / -1.750401 | 0/12 |
| inner_held | context:single1_seed43_controller0_dimensionless | -7.112469 / +3.301778 | -0.226769 / -0.085892 | 1/12 |
| inner_held | context:single1_seed43_controller2_dimensionless | -4.874628 / -2.952126 | -1.013949 / -0.296882 | 3/12 |
| inner_held | context:single2_seed43_controller0_dimensionless | -2.165871 / -0.767621 | -0.441386 / -0.402735 | 0/12 |
| inner_held | context:single2_seed43_controller1_dimensionless | -3.138774 / -1.912913 | -0.014605 / -0.014605 | 0/12 |
| inner_held | seed:17 | -5.342346 / -0.352644 | -1.060998 / -0.602050 | 0/24 |
| inner_held | seed:29 | -5.529434 / -0.286873 | -1.179431 / -0.582405 | 3/24 |
| inner_held | seed:43 | -5.156359 / -0.339089 | -1.157733 / -0.690671 | 1/24 |

## Evidence Boundary

Primary full-roster budget supported: False.
No secondary winner substitutes for hard_record. No new MSE gate or threshold search.
obs8/pred12 rawstride12, pixel/local detector-silver; not metric, seconds, human gold,
true3D, foundation, independent confirmation or submission-readiness evidence.
Independent selection/calibration/confirmation untouched; deployment unchanged.
Stage5C and SMC remain off.
