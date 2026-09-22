# Current Controlled Results

Rebuilt from pinned public aggregates. All policies below use EqMotion ramp candidates.
Four explored SDD sites; three seeds; 8 observed / 12 predicted annotation steps.
Negative easy degradation means improvement. Worst means maximum over all 12 site/seed views.

| Policy | ADE gain % | FDE gain % | Hard gain % | Worst easy degradation % | Switches |
|---|---:|---:|---:|---:|---:|
| Square neural, strict | 3.447 | 4.976 | 3.589 | 1.092 | 37,030 |
| Log neural, strict | 3.398 | 4.900 | 3.675 | 0.911 | 33,793 |
| Square neural, risk rank | 2.600 | 3.825 | 2.585 | 0.107 | 22,539 |
| Log neural, risk rank | 2.807 | 4.122 | 2.971 | 0.164 | 22,539 |
| Forest, risk rank | 3.530 | 5.166 | 3.764 | -0.134 | 22,539 |
| Square neural, gain rank | 3.839 | 5.313 | 5.769 | 2.509 | 22,539 |
| Log neural, gain rank | 3.828 | 5.281 | 5.807 | 2.558 | 22,539 |
| Forest, gain rank | 4.074 | 5.620 | 6.175 | 2.702 | 22,539 |

Counts include repeated query/seed instances, not independent people/events.
Same counts do not imply the same displacement mass or realized harm budget.
Both original primary superiority gates fail; no secondary winner replaces them.

## Paired Development Contrasts

| Preserved comparison | ADE difference (pp) | 95% site-bootstrap CI (pp) |
|---|---:|---:|
| Region minus intermediate (strict) | +0.36871 | [+0.17316, +0.60302] |
| Region minus intermediate (matched count) | -0.04374 | [-0.07376, -0.01372] |
| Adaptive minus frozen region | -0.06749 | [-0.19875, +0.09748] |
| Review minus matched nomination | -2.21244 | [-4.51520, -0.76060] |
| Log minus region | +0.09109 | [-0.00454, +0.21899] |
| Prefix guard minus terminal control | -2.99160 | [-3.70943, -2.41675] |
| Prefix guard minus matched terminal | -0.72655 | [-1.05392, -0.39235] |
| Ramp minus uniform action | -0.24693 | [-0.29914, -0.19472] |
| Forest minus log neural (original primary) | +0.13273 | [-0.06206, +0.48134] |
| Forest minus log neural (same-count risk) | +0.72331 | [+0.61059, +0.82728] |
| Square minus log neural (strict primary) | +0.04951 | [-0.00527, +0.13864] |
| Square minus log neural (same-count risk) | -0.20728 | [-0.25325, -0.14071] |
| Square neural minus forest (same-count risk) | -0.93059 | [-0.99940, -0.85137] |
| Earlier joint minus unary geometry | +0.00000 | [+0.00000, +0.00000] |

The last contrast concerns earlier full-forecast joint controls, not the new ramp support audit.
Bootstrap intervals condition on four development-exposed sites; they are not independent confirmation.

## Joint Support, Not Forecasting Accuracy

| Pool | Queries including seeds | Nonadditive opportunities | Unique opportunity frames | Changed queries including seeds |
|---|---:|---:|---:|---:|
| forest_ratio | 62,796 | 62 | 24 | 3 |
| log_strict | 62,796 | 91 | 52 | 4 |
| square_strict | 62,796 | 113 | 64 | 3 |

No future-target readout or predictive-lift estimate was made in this support audit.
