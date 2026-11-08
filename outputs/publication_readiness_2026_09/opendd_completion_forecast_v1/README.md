# Can Training On Missing-Step Disagreement Improve The Forecast?

I trained the same ego-only comoving EqMotion predictor three ways:
ignore missing future steps, penalize disagreement with constant
velocity (CV) on those steps, or minimize the worse of CV and OLS4
completion regret within each batch. If the new loss helps only by
shrinking edits, a masked model moved to the same peak displacement
should do as well. I tested that control separately from the existing
fixed-budget policy comparison.

This concerns **missing future supervision**, not the missing
observed history studied by [Chib and Singh](https://proceedings.neurips.cc/paper_files/paper/2024/file/e13a3071bd0aeb97ce41b2da921dfdb6-Paper-Datasets_and_Benchmarks_Track.pdf).
On an unannotated step, the largest possible excess Euclidean error
over a given reference is the distance between the two predictions.
The triangle inequality gives the upper bound, and taking the
reference itself as the unknown target attains it. This is not
imputation or a guarantee of recording-level calibration.

## Matched Experiment

The study uses the previously examined [OpenDD](https://arxiv.org/abs/2007.08463)
cohort: 10,475 windows from 202 recordings across seven locations,
with 9,823 complete 12-step futures. I held out each location and
fit on the other six, with three paired seeds and the same 800-update
schedule, batch draws, initialization and model in each comparison.
Each draw can include complete, partial and wholly unobserved future
labels. The model sees eight past observations; future positions and
masks are only used for loss and evaluation. All 21 prediction cells
were frozen before scoring. No thresholds were selected on held data.

| Predictor / training loss | Complete ADE | Endpoint FDE | Full-grid lower gain vs CV, pp [95% interval] | Lower gain vs OLS4, pp | Original risk passes / 606 |
|---|---:|---:|---:|---:|---:|
| CV | 0.5561 | 1.2274 | Reference | — | Not estimable |
| OLS4 | 0.6204 | 1.3195 | — | Reference | — |
| Masked | 0.5611 | 1.2454 | -0.065 [-2.385, +2.156] | +7.437 | 7 |
| CV completion | 0.5580 | 1.2414 | +0.160 [-1.843, +1.787] | +7.678 | 5 |
| Dual completion | 0.5616 | 1.2465 | -0.303 [-1.907, +1.187] | +7.277 | 10 |

ADE and FDE use publisher coordinates and their stated available
subsets. Lower gain charges missing-step disagreement and uses a
common observed CV error denominator; it is not a percent change
in ADE or FDE. Descriptive intervals use 3,000 paired site draws
after averaging seeds. The 606 views are repeated evaluations of
202 recordings, not 606 independent recordings. Baseline ADE/FDE
are from the verified [same-roster readout](../opendd_comoving_eqmotion_v1/README.md).

## Does It Learn A Better Shape?

| Contrast | Direct full-grid lower gain, pp [95% interval] |
|---|---:|
| CV completion vs masked | -0.585 [-1.299, +0.128] |
| Dual completion vs masked | -1.127 [-2.023, -0.141] |
| CV completion vs masked, at identical per-row peaks | -0.451 [-0.881, +0.149] |
| Dual completion vs masked, at identical per-row peaks | -0.933 [-1.656, -0.121] |

Mean peak displacement falls from 0.3785 for masked training to
0.3515 and 0.3525 for CV and dual completion. But matching the peaks
does not establish a direction advantage. For CV completion the
matched contrast is +0.058 pp on fully labeled rows, -0.455 pp on
partial rows and -0.054 pp on wholly unobserved rows. These pieces
sum on the same denominator. The lower score can be negative even
if the unavailable full-label effect is positive.

Mean known easy degradation is 31.64%, 30.62% and 29.68%; known hard
gain is 5.54%, 5.42% and 4.91%. These site means are not substitutes
for the original recording-level risk check. That check retains its
own wholly-unobserved-row debit and support rules, rather than using
the full-grid training objective as a substitute. At the same 1,457 old
edit slots and per-row peaks, the three learned shapes jointly pass
CV/OLS4 and risk in 87, 88 and 80 views; the past-prefix control passes
90. The original shape passes 77. None of the new forecast losses
produces a candidate to advance to DroneCrowd calibration. I kept
the original 2% rule and the 30 independent confirmation groups sealed.

## Runtime And Reproduction

```sh
export OPENBLAS_NUM_THREADS=1
.venv-pytorch/bin/python scripts/run_m3w_opendd_completion_forecast.py
.venv-pytorch/bin/python scripts/run_m3w_opendd_completion_forecast.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_opendd_completion_radius.py
.venv-pytorch/bin/python scripts/build_m3w_opendd_completion_replay.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_completion_forecast.py tests/test_m3w_opendd_comoving_eqmotion.py
```

The arm64 CPU run used four Torch threads, no loader workers,
63 fits and 50,400 updates in 840.93 seconds. The heartbeat and
every-100-update checkpoints are under the ignored
`data/stage_cvpr2027_experiments/opendd_completion_forecast_v1/` directory.
After an interrupted run, verify the old PID has exited and rerun the
same command to resume. Replay matches all forecasts and scores with
zero optimizer updates in 32.29 seconds. Eight focused tests passed;
the legacy full suite was not rerun in the active worktree. CREATE
was queried read-only; no jobs were added or changed. The anonymous
packet needs only NumPy and contains no weights, paths or trajectories.

[Registration](registration.json) | [Forecast freeze](freeze.json) |
[Primary results](analysis.json) | [Same-peak audit](radius_analysis.json) |
[Paper supplement](../evidence_manuscript_v118/README.md)
