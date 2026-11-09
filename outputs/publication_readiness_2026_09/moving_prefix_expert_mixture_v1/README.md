# Short Backtests Improve Ordering, Not Safe Forecast Replacement

I tested whether motion-expert backtests inside eight observed
positions could tell a model which twelve-step trajectory to trust.
This applies established [feature-based forecast averaging](https://robjhyndman.com/publications/fforma/);
it is not a claim to invent expert weighting. The six fixed experts
are CV, anchored OLS4, anchored OLS8, damped OLS4, bounded turn, and
stationary. One learned convex weight vector applies to the whole path.

Each backtest starts from 4, 5, 6, or 7 observed positions and predicts
the rest of the eight. The longer OLS expert uses the available prefix
for that short comparison. Relative errors against OLS4 are features;
none is computed from the requested future. The static, kinematics-only
and prefix arms have the same model, draws and initialization for each
seed. They differ only in which feature blocks are zeroed.

## Source-Excluded Results

Nine fits use the original 69,746 rows in nine fitting groups.
All forecasts for 125,540 exposed A targets were frozen before scoring.
The 34 recordings comprise 14 source groups; three seeds give 102
repeated recording views, not 102 independent recordings. The task
uses eight observed and twelve predicted native annotation steps in
image coordinates, without seconds or metric interpretation.

| Prefix mixture versus | Observed gain (pp) | Full-grid lower gain (pp) | Paired 95% interval for lower gain |
|---|---:|---:|---:|
| Static mixture | +3.093 | -15.300 | [-27.156, -6.425] |
| Kinematics-only mixture | +1.635 | -7.313 | [-13.792, -2.602] |
| CV | +3.445 | -8.915 | [-22.660, -0.334] |
| OLS4 | +6.138 | -24.890 | [-45.280, -10.199] |
| Previous direct kinematic forecaster | -11.894 | -62.199 | [-91.101, -37.204] |
| Kinematics-only, same recipients and peaks | +0.012 | -0.167 | [-0.303, -0.062] |
| OLS4, same recipients and peaks | +0.053 | -0.476 | [-1.188, -0.024] |

All paired gains use a query-balanced, observed CV error denominator.
Full-grid lower scores debit each missing step by forecast
disagreement. The corresponding upper scores are in the readout;
a negative lower score does not prove that the unknown true
effect is negative. Descriptive intervals average seeds and resample
the 14 groups 3,000 times.

The equal-peak control keeps 375,510 edits across repeated seeds,
excluding 1,089 prefix edits without a common nonzero direction.
The support rule does not use future labels. The observed improvement
over the kinematic model mostly vanishes at equal peak. Fully observed
rows contribute only +0.004 pp of that matched gain on the whole-roster
denominator. All three raw mixtures and both matched mixtures pass
0/102 original recording-risk checks. The raw prefix model exceeds
the 2% signed easy limit in 58/102 views; its worst recording p95
known ADE is 311.28 image pixels (kinematics-only: 314.67).

## Why A Backtest Winner Is Not Enough

I used the same 90,965 fully observed targets at every lead. This
diagnostic counts agreement among nontied pairs of analytic experts.
It also computes the cost of picking the single lowest backtest
error expert without learning. That choice is compared to OLS4.

| Lead | Pairwise ordering agreement | Backtest winner gain vs OLS4 (pp) |
|---|---:|---:|
| 1 | 68.72% | +0.093 [-2.722, +3.270] |
| 4 | 72.85% | -0.708 [-3.458, +2.225] |
| 8 | 74.07% | -5.001 [-9.508, -0.953] |
| 12 | 74.00% | -7.441 [-12.333, -2.974] |

The order signal survives, but selecting by it ignores how much a
mistake costs. This is not evidence that short backtests are
uninformative. It is evidence that order accuracy and smaller edits
do not establish a better forecast at the requested horizon. I kept
the original 2% rule and did not open DroneCrowd confirmation.

## Reproduction

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_prefix_expert_mixture.py
.venv-pytorch/bin/python scripts/run_m3w_prefix_expert_mixture.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_prefix_expert_radius.py
.venv-pytorch/bin/python scripts/audit_m3w_prefix_expert_horizons.py
.venv-pytorch/bin/python scripts/build_m3w_prefix_expert_replay.py
```

Native arm64 CPU, four Torch threads, no data-loader workers.
The 9,000 updates and scoring took 26.25 s, with checkpoints every
100 updates. Exact replay used zero updates (14.12 s). Rerunning
the same command after an interruption resumes compatible checkpoints.
Eleven focused tests passed. Only compact results are tracked. The local anonymous archive
recomputes the main table from recording masses; use `replay.py --show`
for its actual table rather than the generic replay banner. No CREATE
job or legacy full-suite run.

[Registration](registration.json) | [Light metrics](readout.json) |
[Lead analysis](lead_alignment.json) |
[Paper supplement](../evidence_manuscript_v120/README.md)
