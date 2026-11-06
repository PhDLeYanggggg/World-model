# Exact-Cohort Fixed-Amplitude Controls

Fresh causal static choices and complete scoring on62,466exposed own-B rows.
All six original coefficients are reported, not selected from this readout.
No new model is trained; frozen learned-policy metrics reproduce exactly.

## Complete Static Matrix

| Policy | Known gain % | Separate lower % | Hard gain % | Easy gain % | Original risk pass | Known easy violations | Easy NA | Worst signed easy degradation % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all_past_eligible_a1 | +1.4339 | +0.3753 | +1.2439 | +0.8483 | 5/24 | 9 | 0 | +8.9535 |
| all_past_eligible_a2 | +2.8100 | +0.6929 | +2.4680 | +0.7136 | 2/24 | 17 | 0 | +24.5552 |
| all_past_eligible_a3 | +5.3862 | +1.1521 | +4.8542 | -1.7840 | 2/24 | 18 | 0 | +64.5794 |
| all_past_eligible_a4 | +9.8136 | +1.3454 | +9.3534 | -12.5123 | 1/24 | 21 | 0 | +165.5308 |
| all_past_eligible_a5 | +15.6137 | -1.3226 | +16.8984 | -45.9265 | 1/24 | 22 | 0 | +395.4346 |
| all_past_eligible_a6 | +12.9387 | -20.9341 | +23.0125 | -136.9229 | 1/24 | 23 | 0 | +874.3682 |
| floor | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 0/24 | 0 | 24 | +0.0000 |
| last_velocity_nonzero_a1 | +1.4344 | +0.3945 | +1.2434 | +0.9255 | 5/24 | 7 | 0 | +8.9535 |
| last_velocity_nonzero_a2 | +2.8114 | +0.7316 | +2.4671 | +0.8869 | 2/24 | 17 | 0 | +24.5552 |
| last_velocity_nonzero_a3 | +5.3899 | +1.2302 | +4.8524 | -1.3941 | 2/24 | 18 | 0 | +64.5794 |
| last_velocity_nonzero_a4 | +9.8229 | +1.5036 | +9.3499 | -11.6400 | 1/24 | 21 | 0 | +165.5308 |
| last_velocity_nonzero_a5 | +15.6379 | -1.0008 | +16.8923 | -43.9871 | 1/24 | 22 | 0 | +395.4346 |
| last_velocity_nonzero_a6 | +13.0010 | -20.2765 | +23.0033 | -132.6690 | 1/24 | 23 | 0 | +874.3682 |

Static policies have24recording views and one frozen forecaster seed17.
They have no cost-head seeds. The floor has zero added motion error but
empty selected-reference risk remains NA, not an identity-certified pass.

## Frozen Learned Controls

| Family | Decision | Known gain % | Separate lower % | Original risk pass | Known easy violations | Easy NA |
|---|---|---:|---:|---:|---:|---:|
| expert_context | independent | +0.2650 | +0.2141 | 28/72 | 19 | 8 |
| expert_context | joint_free | +0.7906 | +0.7116 | 24/72 | 26 | 6 |
| frozen | independent | +0.2255 | +0.1817 | 28/72 | 13 | 7 |
| frozen | joint_free | +0.6675 | +0.5873 | 23/72 | 27 | 7 |
| history_refit | independent | +0.1979 | +0.1604 | 29/72 | 16 | 6 |
| history_refit | joint_free | +0.6149 | +0.5412 | 25/72 | 23 | 6 |

72learned views repeat24recordings over3risk-head seeds. They are not
72independent recordings or three forecaster runs. All five decisions
per learned family are retained in analysis.json and the per-context metrics.

## Registered Primary

Expert-context joint-free minus guarded alpha1/32; all required sources:
| Contrast | Point pp | Exploratory source interval |
|---|---:|---|
| completion_lower_pp | -1.6969 | [-3.1268, -0.3972] |
| completion_upper_pp | +0.4093 | [-0.2709, +1.2253] |
| known_pp | -0.6438 | [-1.2520, +0.0489] |

Eight source components,3,000resamples,with learned risk seeds averaged
within source. The static baseline is repeated only for pairing, not as
new independent model runs. Separate completion lower gains cannot simply
be subtracted to get the paired contrast. A negative lower contrast is a
failed conservative criterion, not proof of the unknown true effect sign.

## Verdict

Learned primary positive: False.
Any fixed nonfloor policy passes all24original views: False.
No coefficient or deployment winner is selected. Average accuracy does
not override original per-record2%positive all/easy harm,signed easy-net,
reference support or unknown-label completion. The original24-source-group
main stays NA; no new independent C/H is opened.

Native8/12available-label image-coordinate ADE is not complete12ADE,
SDD raw t50,metric/seconds,true3D,foundation or human-gold evidence.
Annotation-generation causality remains unverified. Stage5C/SMC off.

[Protocol](protocol.md) | [Interpretation](failure_analysis.md) | [Reproduction](operation_zh.md)
