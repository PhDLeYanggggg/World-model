# Whole-Recording Policy Readout

## Material Passport

Fresh_run:36 neural policy fits,72,000 updates;training 1043.169s;readout 96.813s.
Training peak RSS 3,130,507,264 bytes. All finals frozen before held scoring.
Cached_verified: source data,forecast/risk models,original and stochastic policies.
Not_run: independent test,new forecaster,deployment. Head seeds are not new forecasting seeds.
Held recordings were exposed historically. This is exploratory,not whole-pipeline OOF.

## Full Policy Safety

| Cohort | Policy | Selected | Unknown | Budget views | Budget recording views | Known all/easy violations | Easy degradation violations | Undefined easy |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| fit | floor | 0 | 0 | 0/72 | 0/444 | 0/0 | 0 | 72 |
| fit | origin | 62902 | 1567 | 1/72 | 16/444 | 35/25 | 68 | 7 |
| fit | stochastic | 46508 | 226 | 35/72 | 125/444 | 28/5 | 0 | 9 |
| fit | exact_dual | 43476 | 180 | 33/72 | 120/444 | 31/5 | 0 | 9 |
| fit | exact_gradient | 138260 | 228 | 53/72 | 178/444 | 13/0 | 0 | 6 |
| inner_held | floor | 0 | 0 | 0/72 | 0/186 | 0/0 | 0 | 72 |
| inner_held | origin | 32621 | 847 | 0/72 | 3/186 | 61/44 | 61 | 14 |
| inner_held | stochastic | 19915 | 326 | 4/72 | 17/186 | 43/50 | 27 | 11 |
| inner_held | exact_dual | 18119 | 304 | 2/72 | 18/186 | 45/44 | 25 | 14 |
| inner_held | exact_gradient | 56720 | 869 | 0/72 | 2/186 | 64/60 | 49 | 3 |

Original2% all/easy positive harm/reference and easy-preservation requirements unchanged.
Undefined selected risk does not pass. Floor selects nobody; no fictitious safe zero ratio.

## Paired Utility

| Cohort | Contrast | Full lower/upper pp | Same-count lower/upper pp | Empty matched |
|---|---|---:|---:|---:|
| fit | exact_gradient_minus_origin | -7.261186 / +1.228389 | -2.538256 / -1.039715 | 3 |
| fit | exact_dual_minus_stochastic | -0.072365 / -0.071681 | -0.038079 / -0.037911 | 5 |
| fit | exact_gradient_minus_stochastic | +2.981433 / +2.982839 | +0.302219 / +0.302864 | 3 |
| fit | exact_gradient_minus_exact_dual | +3.053565 / +3.054754 | +0.351603 / +0.352106 | 4 |
| inner_held | exact_gradient_minus_origin | -4.226985 / +0.851822 | -1.517747 / -0.628100 | 7 |
| inner_held | exact_dual_minus_stochastic | -0.143620 / +0.033310 | -0.050685 / -0.028776 | 11 |
| inner_held | exact_gradient_minus_stochastic | +0.377395 / +1.916356 | +0.073468 / +0.171986 | 9 |
| inner_held | exact_gradient_minus_exact_dual | +0.450563 / +1.953498 | +0.119686 / +0.196982 | 6 |

Deterministic partial-label completion bounds,not sampling confidence intervals or relative FDE gains.
Full known reference denominator in each view,then equal-view averaging. Count matching is utility only.
No posthoc policy selection; exact_gradient remains primary.

## Compute and Exposure

```json
{
  "exact_dual": {
    "fits": 18,
    "optimizer_steps": 36000,
    "seconds": 423.7248531678342,
    "sampled_primal_rows": 56832000,
    "whole_record_forward_rows": 1789950000,
    "full_gradient_rows": 0,
    "final_preupdate_positive_record_channels": 287,
    "final_duals_at_cap": 32,
    "final_dual_count": 888,
    "mean_final_full_objective": -0.2056464178657633,
    "mean_final_soft_hard_gap": 0.005482431974248341
  },
  "exact_gradient": {
    "fits": 18,
    "optimizer_steps": 36000,
    "seconds": 542.6211402499466,
    "sampled_primal_rows": 0,
    "whole_record_forward_rows": 1789950000,
    "full_gradient_rows": 1789950000,
    "final_preupdate_positive_record_channels": 284,
    "final_duals_at_cap": 20,
    "final_dual_count": 888,
    "mean_final_full_objective": -0.22744726210294175,
    "mean_final_soft_hard_gap": 0.007671743021207055
  }
}
```
Exact_dual matches original sampled primal draws. Exact_gradient has greater deterministic gradient
exposure; both see every eligible TRAIN row for dual updates. Update-matched,not compute-matched.

Final trace constraints are from the pre-update model,not the final frozen readout.
The dual-weighted objective changes during training; it is not a fixed validation loss.

## Benefit and Harm Retention

| Cohort | Policy | Known benefit retained | Known harm retained | Known net change pp |
|---|---|---:|---:|---:|
| fit | stochastic | +0.240993 | +0.205313 | -5.998535 |
| fit | exact_dual | +0.233638 | +0.218142 | -6.070558 |
| fit | exact_gradient | +0.593556 | +0.285273 | -3.016398 |
| inner_held | stochastic | +0.208999 | +0.200534 | -2.834457 |
| inner_held | exact_dual | +0.202502 | +0.215893 | -2.889612 |
| inner_held | exact_gradient | +0.529632 | +0.526026 | -1.687582 |

Fractions use equal-view normalized known mass relative to the original frozen policy.
Missing outcomes remain in paired completion bounds; they are not counted as zero harm.

## Complete Slices

| Cohort | Slice | Primary vs original full pp | Primary vs stochastic full pp | Primary budget views |
|---|---|---:|---:|---:|
| fit | source:eu-locality-007 | -8.164849 / +0.547896 | +5.341443 / +5.341443 | 6/6 |
| fit | source:eu-locality-008 | -0.908997 / +0.809491 | +1.321495 / +1.335428 | 3/6 |
| fit | source:eu-locality-020 | -3.063201 / +3.070144 | +1.999247 / +2.000247 | 6/6 |
| fit | source:eu-locality-048 | -10.357167 / -2.913520 | +1.927301 / +1.927301 | 6/6 |
| fit | source:eu-locality-067 | -2.584551 / -0.780104 | +1.849104 / +1.849574 | 6/6 |
| fit | source:eu-locality-074 | -2.852154 / -1.117058 | +0.568117 / +0.569585 | 0/6 |
| fit | source:eu-locality-082 | -25.185740 / +20.548642 | +1.454503 / +1.454503 | 0/6 |
| fit | source:eu-locality-110 | -2.337115 / -0.745211 | +6.393817 / +6.393817 | 5/6 |
| fit | source:eu-locality-112 | -7.975878 / +4.967987 | +2.159214 / +2.159214 | 6/6 |
| fit | source:eu-locality-119 | -3.792777 / +0.363596 | +4.977104 / +4.977104 | 5/6 |
| fit | source:eu-locality-124 | -11.535952 / -5.631394 | +2.364038 / +2.364038 | 4/6 |
| fit | source:eu-locality-126 | -8.375846 / -4.379806 | +5.421816 / +5.421816 | 6/6 |
| fit | context:single0_seed43_controller1_dimensionless | -5.252444 / -0.217003 | +1.940524 / +1.946215 | 12/12 |
| fit | context:single0_seed43_controller2_dimensionless | -7.474894 / -1.562190 | +4.636713 / +4.636713 | 10/12 |
| fit | context:single1_seed43_controller0_dimensionless | -3.204514 / +0.310009 | +1.608049 / +1.610060 | 9/12 |
| fit | context:single1_seed43_controller2_dimensionless | -5.486684 / +0.733416 | +3.099447 / +3.100181 | 6/12 |
| fit | context:single2_seed43_controller0_dimensionless | -5.440452 / -1.170367 | +4.901488 / +4.901488 | 10/12 |
| fit | context:single2_seed43_controller1_dimensionless | -16.708125 / +9.276467 | +1.702378 / +1.702378 | 6/12 |
| fit | seed:17 | -7.654044 / +1.491824 | +2.803355 / +2.804987 | 18/24 |
| fit | seed:29 | -7.861533 / +1.409109 | +3.213185 / +3.214724 | 17/24 |
| fit | seed:43 | -6.267980 / +0.784233 | +2.927760 / +2.928807 | 18/24 |
| inner_held | source:eu-locality-007 | -0.417627 / -0.232649 | +0.080534 / +0.229094 | 0/6 |
| inner_held | source:eu-locality-008 | -1.594776 / +0.656072 | +0.491152 / +0.803693 | 0/6 |
| inner_held | source:eu-locality-020 | -15.945862 / +15.816747 | -4.119693 / +7.952989 | 0/6 |
| inner_held | source:eu-locality-048 | -4.058248 / -0.465327 | +0.867947 / +1.302541 | 0/6 |
| inner_held | source:eu-locality-067 | -4.344423 / +0.442346 | +0.976071 / +1.340672 | 0/6 |
| inner_held | source:eu-locality-074 | -2.942235 / -1.558407 | +0.325560 / +0.567502 | 0/6 |
| inner_held | source:eu-locality-082 | -0.585562 / +1.347992 | -0.365403 / +0.352927 | 0/6 |
| inner_held | source:eu-locality-110 | -1.401119 / +0.051153 | +3.343985 / +4.419891 | 0/6 |
| inner_held | source:eu-locality-112 | -1.737647 / +1.412906 | +0.966587 / +1.247535 | 0/6 |
| inner_held | source:eu-locality-119 | -2.094012 / +0.483463 | +1.740370 / +3.372500 | 0/6 |
| inner_held | source:eu-locality-124 | -5.901425 / +1.018018 | +0.147690 / +1.332990 | 0/6 |
| inner_held | source:eu-locality-126 | -9.700885 / -8.750454 | +0.073941 / +0.073941 | 0/6 |
| inner_held | context:single0_seed43_controller1_dimensionless | -6.555880 / +4.096799 | +0.029828 / +3.254821 | 0/12 |
| inner_held | context:single0_seed43_controller2_dimensionless | -3.461932 / +0.309263 | +2.088263 / +3.406203 | 0/12 |
| inner_held | context:single1_seed43_controller0_dimensionless | -6.415774 / +4.128121 | -0.922089 / +2.445127 | 0/12 |
| inner_held | context:single1_seed43_controller2_dimensionless | -4.634815 / -2.209120 | +0.218754 / +0.764065 | 0/12 |
| inner_held | context:single2_seed43_controller0_dimensionless | -1.445159 / +0.350730 | +0.568027 / +1.271035 | 0/12 |
| inner_held | context:single2_seed43_controller1_dimensionless | -2.848350 / -1.564861 | +0.281588 / +0.356887 | 0/12 |
| inner_held | seed:17 | -4.068873 / +0.643264 | +0.290358 / +1.979023 | 0/24 |
| inner_held | seed:29 | -4.378308 / +1.103795 | +0.572245 / +1.969549 | 0/24 |
| inner_held | seed:43 | -4.233775 / +0.808406 | +0.269582 / +1.800497 | 0/24 |

Record/row views repeat across contexts/seeds; not independent samples.
Primary full budget supported: False.
No independent confirmation or deployment change. Empirical feasibility is not population safety.
obs8/pred12 rawstride12,pixel/local detector-silver;no metric,seconds,human gold,true3D,foundation claim.
Stage5C/SMC remain off.
