# DroneCrowd Calibration Outcome

**No registered threshold qualifies.** Independent confirmation remains sealed.

40 recordings, 15 exclusion groups, three frozen seeds and 120 repeated recording-seed views.
The 3,000 paired bootstrap draws resample groups after seed averaging.
This is calibration, not independent confirmation or deployment evidence.

| Arm | Margin | Switches | Risk passes / 120 | Undefined easy | All harm >2% | Signed easy >2% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| constant_velocity_causal_fd | NA | 0 | 0 | 120 | 0 | 0 |
| history_ols4_velocity | NA | 3,980,949 | 24 | 0 | 96 | 0 |
| independent | NA | 67,440 | 18 | 0 | 102 | 1 |
| scene | NA | 67,440 | 22 | 0 | 98 | 1 |
| pair_additive | NA | 67,440 | 67 | 0 | 53 | 0 |
| joint | NA | 67,440 | 69 | 0 | 50 | 0 |
| gate_0 | 0.00 | 362 | 7 | 99 | 14 | 0 |
| gate_1 | 0.01 | 362 | 7 | 99 | 14 | 0 |
| gate_2 | 0.03 | 362 | 7 | 99 | 14 | 0 |
| gate_3 | 0.10 | 362 | 7 | 99 | 14 | 0 |

Risk-diagnostic counts can overlap. An undefined selected reference is not a pass.

| Arm | Conservative gain vs CV, pp [95% CI] | Conservative gain vs OLS4, pp [95% CI] | Eligibility |
| --- | ---: | ---: | --- |
| constant_velocity_causal_fd | NA | NA | control |
| history_ols4_velocity | +51.288 [+46.487, +55.878] | NA | control |
| independent | +0.445 [+0.292, +0.582] | -51.096 [-55.653, -46.368] | control |
| scene | +0.476 [+0.322, +0.621] | -51.067 [-55.625, -46.331] | control |
| pair_additive | +1.748 [+1.464, +2.083] | -49.797 [-54.483, -44.960] | control |
| joint | +1.766 [+1.477, +2.106] | -49.780 [-54.471, -44.945] | control |
| gate_0 | +0.010 [+0.000, +0.030] | -51.537 [-56.159, -46.686] | recording_risk_or_support, cv_lower_ci_not_positive |
| gate_1 | +0.010 [+0.000, +0.030] | -51.537 [-56.159, -46.686] | recording_risk_or_support, cv_lower_ci_not_positive |
| gate_2 | +0.010 [+0.000, +0.030] | -51.537 [-56.159, -46.686] | recording_risk_or_support, cv_lower_ci_not_positive |
| gate_3 | +0.010 [+0.000, +0.030] | -51.537 [-56.159, -46.686] | recording_risk_or_support, cv_lower_ci_not_positive |

Gains use the observed CV error mass as denominator. Distances are image pixels;
8/12 refers to annotation steps. OLS4 is not a calibration gate.
