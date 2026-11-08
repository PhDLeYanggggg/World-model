# Standalone Hindcast: Opened-Clip Risk Pass, Fitting-Source Failure

**Fresh deterministic experiment.** The previous matched backtest borrowed
its intervention counts from a learned policy. Here the policy supplies
its own recipients and amplitudes, using only the eight observed positions.
We retain the 12-step horizon, image-coordinate units, original CV/OLS4
paths, fitting-derived easy threshold, and original 2% risk rule.
This is a diagnostic on opened calibration clips, not a new independent
test, a deployment decision, or a revision to the failed registered
calibration. The 30 independent confirmation source groups stay sealed.

## Fitting and Holdout Selection

For each possible blend (0.25, 0.50, 0.75, 1.00), compare it to CV at
four origins within the eight observations, dividing each error
difference by mean lead so the shortest origin does not automatically
have the smallest scale. Four fixed guards (no guard, positive mean,
positive mean and 3/4 agreement, 4/4 agreement) make 16 candidates.
Among the nine original fitting groups (24 recordings, 69,746 rows),
select by worst-group then mean risk pass fraction, breaking ties by
mean conservative gain. The fit tie-break is row-weighted; external
utility uses the original query weighting. No calibration labels
enter selection.

No candidate passes any fitting recording. The leave-one-group-out
procedure also passes **0/24**, using a choice made on the other eight
groups for each held-out group. The all-fitting choice is a 0.50 blend
when at least three origins favor the blend and mean gain is positive.
On fitting data, **22/24** recordings already exceed a risk threshold
on released labels alone. The support failure is not explained only by
the conservative debit on missing labels.

## Opened DroneCrowd Calibration Recordings

| Deterministic policy | Switch rate | Conservative gain vs CV, pp [95% source interval] | Original risk passes |
|---|---:|---:|---:|
| Full OLS4 | 75.50% | +51.288 [+46.487, +55.878] | 8/40 |
| Fixed half | 75.50% | +32.643 [+30.303, +34.838] | 34/40 |
| Standalone three-origin gate | 60.61% | +26.477 [+23.517, +29.531] | 40/40 |
| Latest-origin rank at matched count | 60.61% | +26.533 [+23.025, +29.991] | 35/40 |

These are **40 distinct recording views**, not three repetitions of
a deterministic policy. The two matched arms make 1,065,288 changes
at exactly the same counts and amplitudes in each of 11,240 queries.
Five passes differ across four source groups. Equal-source paired
pass-rate difference is +21.48 percentage points (3,000-draw grouped
95% interval [+2.96, +42.96]); the unweighted recording difference
is +12.5 points. Both intervals and point estimates describe this
opened roster, not independent generalization. The direct conservative
accuracy difference against the matched control is -0.062 pp
[-0.688, +0.600]. Against full OLS4 it is -24.954 pp
[-26.766, -22.966].

The standalone arm's worst actual all/easy positive-harm ratios are
1.9834% and 1.9955%. The easy ratio is only 0.0045 percentage points
below the limit. Signed easy degradation is a separate statistic.
The unchanged CV-identity policy has undefined selected denominators;
it is not counted as a pass. This comparison supports a local
recipient-selection effect on risk, not better average accuracy,
transferable safety, or a neural contribution. Strong motion
controls are also motivated by [Scholler et al.](https://arxiv.org/abs/1903.07933).

**Next:** diagnose why motion smoothing helps in the opened release
but the identical policy harms original sources with known labels.
Keep OLS4 as the accuracy benchmark and the CV-based rule as the
unchanged risk criterion. Do not tune on the five failed controls
or spend the sealed confirmation on resolving this discrepancy.

## Replay

~~~sh
p=scripts/run_m3w_standalone_hindcast.py
for phase in register fit infer score replay; do .venv-pytorch/bin/python "$p" "$phase"; done
.venv-pytorch/bin/python scripts/audit_m3w_standalone_hindcast.py
~~~

Ran locally on native arm64 with no new neural fits or loader workers.
Inference and scoring save atomic per-recording artifacts and heartbeat;
replay matched the aggregate without reopening calibration labels.
Fifteen focused tests passed. No raw data, row caches, or weights are
part of this report. Stage5C and SMC remain disabled.
[Fitting evidence](fitting.json) | [Frozen roster](freeze.json) |
[Results](analysis.json) | [Risk gap audit](risk_gap.json).
