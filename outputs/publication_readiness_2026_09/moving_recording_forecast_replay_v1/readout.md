# Forecast Accuracy Is Not A Recording-Level Risk Pass

I re-evaluated the complete frozen A roster recording by recording with the
original risk scorer. It includes 125,540 rows in 1,613 queries from 34 distinct
recordings and 14 sources, repeated for three actual training seeds. All six
causal baselines, PCT and EqMotion, and all five already frozen switch policies
remain. The recording-local sums reproduce the earlier learned-anchor paired
source contrasts to within 6.7e-16 percentage points. No training or forecast
inference is repeated, and no data or threshold is selected from this readout.

## Accuracy Against The Fitting-Selected Baseline

Both frozen predictor families selected OLS4 on their fitting sources. I keep
OLS4 as an accuracy comparator, and keep the original CV risk floor. The table
uses the registered query-balanced source summaries and 3,000 source draws
after seed averaging. The sources are exposed development; these intervals
are exploratory, not independent confirmation.

| Candidate minus comparator | Known utility (pp) | Paired completion lower (pp; 95% source interval) |
|---|---:|---:|
| OLS4 vs CV | -3.959 | -17.742 [-33.413, -6.684] |
| PCT vs OLS4 | -21.505 | -70.168 [-119.306, -26.577] |
| EqMotion vs OLS4 | +20.472 | -1.503 [-21.072, +14.223] |
| EqMotion vs PCT | +43.663 | -11.967 [-64.182, +31.584] |
| Original fixed-count selection vs OLS4 | +4.533 | -10.143 [-23.034, -0.505] |
| Learned anchor vs OLS4 | +4.558 | -10.074 [-22.916, -0.512] |
| Learned anchor vs original fixed-count selection | +0.021 | -0.218 [-0.476, -0.045] |

Each gain uses the comparator named in the protocol for normalization: OLS4
for comparisons against OLS4, CV for EqMotion-vs-PCT and the policy contrast.
These are neither differences of ratios with unlike denominators nor errors
in metric units. The lower bound subtracts trajectory disagreement for
wholly unobserved rows; it does not impute actual errors or fill partial labels.

## Unchanged Whole-Recording Contract

| Policy | Pass / 102 views | All completion risk >2% | Easy completion risk >2% | Easy completion risk undefined | Signed easy degradation >2% | Worst recording p95 ADE (px) |
|---|---:|---:|---:|---:|---:|---:|
| CV, no intervention | NA | 0 | 0 | 102 | 0 | 288.06 |
| OLS4 | 0 | 102 | 102 | 0 | 90 | 335.11 |
| PCT | 0 | 99 | 102 | 0 | 84 | 605.15 |
| EqMotion | 0 | 99 | 102 | 0 | 83 | 360.14 |
| Original fixed-count selection | 0 | 92 | 84 | 18 | 38 | 288.10 |
| Dispersion | 0 | 97 | 86 | 16 | 39 | 288.06 |
| Visibility weight only | 0 | 96 | 86 | 16 | 38 | 288.60 |
| Constant anchor | 0 | 94 | 83 | 19 | 39 | 288.10 |
| Learned anchor | 0 | 95 | 85 | 17 | 41 | 288.06 |

These are 102 repeated views, not 102 independent recordings. Positive-harm
ratios use the original selected known reference mass. This is why the
earlier all-easy positive-harm diagnostic cannot be used as the operational
2% gate. Signed easy net uses all easy rows. CV's no-intervention risk ratios
are undefined, not a newly certified pass. Equal switch counts do not equalize
risk; the revised model retains the worst recording tail and passes no view.
Endpoint FDE, worst recording ADE, full baseline table, unobserved rows, raw
masses and every per-record risk remain in [the scalar archive](analysis.json).

## Reproduction

```sh
.venv-pytorch/bin/python scripts/run_m3w_recording_forecast_replay.py run
.venv-pytorch/bin/python scripts/run_m3w_recording_forecast_replay.py replay
```

Definitions fixed at `34e7ef30`. Both invocations exit 0, produce exactly the
same aggregate and record-level JSON, and take 14.91/14.63 s on the native
arm64 environment. The small scorer now consumes one recording at a time;
the A verification still reads the existing fold-level forecast caches.
This is `fresh_run` scoring of `cached_verified` paths and actions. New model
training, external roles, independent outcomes and deployment are `not_run`.
No Stage5C/SMC, metric or seconds claim. The paper still needs the independently
calibrated confirmation study and a method that beats the original rule.
