# Accounting For Missing Outcomes Narrows, But Does Not Reverse, The Loss

I fitted 42 source-excluded classifiers to predict whether a future trajectory
would be wholly unobserved. I kept the original three EqMotion seed paths and
gain/harm models fixed, then used the estimated missingness to price departures
from the single-seed control. Every arm makes 3,225 interventions per seed
across the same 1,613 queries. All 125,540 rows and 14 sources are retained.

| Compared with the original single-seed control | Known utility difference (pp) | Completion lower (pp; 95% source interval) | Easy positive harm change (pp) | Changed decisions / wholly unknown changed across seeds |
|---|---:|---:|---:|---:|
| Unadjusted dispersion | +0.049 | -3.535 [-8.981, -0.356] | -1.722 | 6,348 / 293 |
| Visibility weight only | +0.064 | -3.886 [-10.158, -0.365] | -1.593 | 6,530 / 306 |
| Anchor with constant fitting rate | -0.015 | -1.896 [-5.450, -0.078] | -1.668 | 1,378 / 69 |
| Anchor with learned missingness (primary) | +0.021 | -0.218 [-0.476, -0.045] | -1.576 | 2,852 / 66 |

**The component fails.** The primary known-label interval is
[-0.019, +0.072] pp and the easy-harm-change interval is [-5.458, +0.694] pp.
The three seed means for the completion lower are -0.283, -0.221 and -0.149 pp.
Intervals resample the 14 sources after averaging seeds, not 376,620 repeated
agent-window rows. Table columns compare each arm to the same reference; their
completion bounds must not be subtracted to claim a direct treatment effect.

The learned anchor reduces the number of altered unknown rows versus the
unadjusted policy, and leaves much less of the paired contrast to the missing
label bound. The bound still outweighs the small known utility difference.
An estimate of average missingness alone also helps preserve reference
choices, so this does not establish that the learned classifier generalizes.
Brier error is 0.037703 versus 0.037785 for the fitting-rate comparator;
the difference interval [-0.002727, +0.003056] crosses zero. The mean easy
harm change is not a per-recording result: 43 of the 102 repeated recording
views exceed 2% known easy positive harm versus 41 for the original control.
Only 34 distinct recordings contribute. No full operational risk pass.

## Reproduction

```sh
.venv-pytorch/bin/python scripts/run_m3w_support_anchored_cost.py fit
.venv-pytorch/bin/python scripts/run_m3w_support_anchored_cost.py score
.venv-pytorch/bin/python scripts/run_m3w_support_anchored_cost.py replay
```

Registration `2e81d65b` precedes fitting. PIDs 11356/11399/11428 all exit 0.
Fit 18.22 s, score 6.50 s, replay 6.54 s, about 2.04 GB peak fit RSS. Per-cell
checkpoints allow the same fit to resume; all actions freeze before scoring.
Replay reproduces all 42 probability arrays and all actions exactly, including
the old dispersion control and the complete saved scalar results. Forecasters
and costs are cached verified, support fits and scores are fresh. No new
forecaster training, external outcomes, full 2% gate or changed thresholds.

Do not deploy this model or treat the narrower lower bound as an accuracy gain.
Further same-source tuning would not supply independent evidence. The next
decisive test needs separately assigned calibration/confirmation data and a
frozen comparison, retaining the original support and easy-harm contract.
Native image coordinates and 8/12 annotation steps only; Stage5C and SMC off.

[Registered design](protocol.md) | [All source/recording results](analysis.json)
