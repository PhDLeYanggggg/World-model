# Cost-Moment Cross-Fit Results

Source: fresh_run for144 controller heads and diagnostics; cached_verified for the frozen
forecaster bank. This is three-source fitting versus one inner-held source, not independent
confirmation or the previous complete selection policy. Outer readout outcomes are unused.

## Primary Reference-Cost Diagnostic

MSE skill is100*(1-head MSE/fitting-constant MSE); positive is better. Each locality first
averages its dependent role/seed views. Intervals bootstrap12localities with3000 draws.

| Candidate | Fitting reference skill % | Held reference skill % | Held minus fit points |
|---|---:|---:|---:|
| dimensionless | 45.3679 [42.2123, 48.3434] | 13.4053 [3.7459, 22.1429] | -31.9626 [-42.0507, -21.9147] |
| damped | 43.4875 [40.3607, 46.5703] | 9.7190 [-3.5613, 20.5859] | -33.7686 [-46.8214, -21.7875] |

## Harm and Fixed2% Screen

| Candidate | Fitting harm skill % | Held harm skill % | Held screen rate | Held actual harm/reference |
|---|---:|---:|---:|---:|
| dimensionless | 31.0078 [28.9481, 33.1854] | 14.8819 [10.7440, 19.0080] | 0.2198 [0.1772, 0.2673] | 0.0487 [0.0319, 0.0692] |
| damped | 19.9756 [17.5661, 22.4140] | 8.7336 [3.1006, 14.1963] | 0.5508 [0.4997, 0.6085] | 0.0260 [0.0097, 0.0538] |

| Candidate | Fit predicted ratio | Fit actual ratio | Held predicted ratio | Held actual ratio |
|---|---:|---:|---:|---:|
| dimensionless | 0.0102 [0.0097, 0.0106] | 0.0285 [0.0238, 0.0329] | 0.0100 [0.0094, 0.0105] | 0.0487 [0.0319, 0.0692] |
| damped | 0.0076 [0.0069, 0.0082] | 0.0102 [0.0091, 0.0116] | 0.0075 [0.0067, 0.0083] | 0.0260 [0.0097, 0.0538] |

The all-risk screen omits utility/easy heads on purpose, preventing the other heads
from seeing the held controller source. It is not a proposed deployable policy. An empirical
ratio over2% diagnoses a miscalibrated predicted screen; it does not invalidate a nonexistent
conformal guarantee. A zero true reference denominator remains undefined, with absolute harm retained.

## Numerator Versus Denominator

Actual/predicted reference <1 means an inflated predicted risk budget. Actual/predicted harm
>1 means underestimated harm. These are equal-locality means of dependent-view ratios, not
a pooled probability or a factorization using independent medians.

| Candidate | Held reference actual/pred | Held harm actual/pred | Screen reference actual/pred | Screen harm actual/pred |
|---|---:|---:|---:|---:|
| dimensionless | 0.9840 [0.7964, 1.1669] | 1.1892 [0.9889, 1.4737] | 0.4192 [0.3520, 0.4876] | 1.8045 [1.3498, 2.2839] |
| damped | 0.9877 [0.7875, 1.1815] | 1.2898 [0.9120, 1.8143] | 0.6909 [0.5569, 0.8285] | 1.5569 [0.9182, 2.4477] |

## Seed Breakdown

| Candidate | Seed | Held reference skill % | Held harm skill % |
|---|---:|---:|---:|
| dimensionless | 17 | 11.8422 [-0.8022, 21.8989] | 15.3362 [11.2485, 19.3019] |
| dimensionless | 29 | 12.0065 [0.6152, 22.0385] | 13.9793 [9.4680, 18.2804] |
| dimensionless | 43 | 16.3672 [9.8096, 23.3170] | 15.3302 [10.6128, 20.0018] |
| damped | 17 | 7.4005 [-10.8061, 20.4194] | 8.1836 [2.5483, 13.8027] |
| damped | 29 | 8.2193 [-5.8215, 20.1265] | 9.7382 [4.6140, 14.8984] |
| damped | 43 | 13.5371 [4.9241, 21.6737] | 8.2791 [1.5894, 14.3372] |

## Scope

No trajectory deployment gain is estimated by this experiment. No model/threshold was selected
from these held diagnostics. Unknown labels are excluded, zero-CV cases retained. Per-locality
fixed score bins and absolute costs are preserved privately; public summaries contain aggregates.
Independent selection/calibration/confirmation remain closed. Detector-derived silver image-local
tracks; obs8/pred12 at raw-frame stride12. No metric, seconds, human-gold, true3D or foundation
claim. No Stage5C execution, SMC or deployment change.
