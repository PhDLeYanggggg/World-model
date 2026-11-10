# Does donor calibration make angular changes worthwhile?

I cross-fitted the rotation threshold on the six donor locations
for each held location and seed. Each calibration predictor and
its nominator exclude both that outer location and its own
calibration location. I froze the 21 threshold decisions at
`1d285ab78` before scoring the outer locations. Twelve choose a
finite mean threshold; none meets the stronger all-donor condition.
The outer score reuses the already frozen six-donor factored model.
The use of five-donor heads for threshold fitting is explicit;
this is a nested development test, not a coverage guarantee.

| Rule on the original edited rows | Rotations / 780 | Dual passes / 606 | Risk passes | Direct lower gain vs incumbent, pp [95% location interval] |
| --- | ---: | ---: | ---: | ---: |
| Keep incumbent | 0 | 60 | 88 | Reference |
| Zero threshold, learned angles | 710 | 70 | 98 | -0.00893 [-0.02838, +0.00371] |
| Mean calibrated, learned angles | 198 | 58 | 86 | -0.00636 [-0.01752, +0.00056] |
| Stable calibrated, learned angles | 0 | 60 | 88 | 0 |
| Donor constant angle on the mean mask | 198 | 65 | 92 | +0.00614 [+0.00066, +0.01583] |

Calibration suppresses many weak predictions, but the learned
angle choice does not improve on keeping the original. The
simple constant-angle control has a small positive development
result on the same rows. The [matched mask audit](../opendd_angular_calibration_mask_v1/README.md)
checks whether that result comes from the angle or the recipient
choice. No arm resolves the 397 undefined risk views, and no
policy is released. All arms keep the same 780 edited recipients,
native 8/12 windows, per-step norms, and original 2% rule.

Sixty-three inner fits, 25,200 updates in 30.7 seconds on native
arm64. Zero-update replay reproduced the freeze in 10.7 seconds.
No optimizer steps occur in the outer score. Intervals
use 3,000 location draws after averaging seeds. No external
outcomes or held thresholds were changed.

[Protocol](protocol.md) | [Pre-score freeze](calibration_freeze.json) | [Held readout](readout.json)

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_opendd_angular_calibration.py calibrate
.venv-pytorch/bin/python scripts/run_m3w_opendd_angular_calibration.py score
```

Use `calibrate --replay` to require completed checkpoints and
verify the freeze without updates. Training saves a heartbeat
and a checkpoint every 100 updates; it uses no data workers.
