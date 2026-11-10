# OpenDD: can a support score improve release without changing the forecasts?

I kept the existing joint proposal for each recording and changed only its
release score. The two edited actors, their paths, their displacement from
CV, and all other forecasts are identical across arms when they choose the
same recording. The original proposal and the existing actor support head
exclude both the held location and the separate calibration location.

The 126 held directions are repeated views of seven development locations,
not 126 independent sites. The main comparison takes the same 2,196 selected
recording views (4,392 forecast edits) per arm. The paired 95% intervals
average the three seeds and repeated calibration directions within location
and resample seven locations 3,000 times.

| Ordering | Matched selected views | Original risk passes | Risk and both accuracy passes | Undefined selected-easy risk |
|---|---:|---:|---:|---:|
| Original merit | 2,196 | 880 | 710 | 1,118 |
| Merit plus log predicted support | 2,196 | 842 | 752 | 1,108 |
| Predicted support alone | 2,196 | 923 | 815 | 902 |

Versus original merit at the same count, adding support changes the
location-weighted dual-pass rate by **+1.93 pp [-0.67, +4.33]**. It reduces
the conservative trajectory gain by **0.001029 pp** (signed contrast
**[-0.001652, -0.000437]**), and reduces conservative relative-motion gain by
**0.01066 native units [-0.01690, -0.00565]**. Its observed-label trajectory
contrast is +0.000272 pp [-0.000361, +0.001169]. The paired missing-label
debit of 0.001302 pp explains why observed gain and the conservative
contrast point in opposite directions. Support alone has the same pattern:
more counted successes but a worse conservative trajectory contrast.

The missing-easy count pools repeated recordings; after giving locations
equal weight the combined score's undefined-rate difference is **+0.96 pp
[-4.77, +7.21]**, not a reliable reduction. This distinction matters because
the same handful of larger locations contributes many more views.

| Ordering | Calibrated held releases | Original risk passes | Risk and both accuracy passes | Undefined selected-easy risk |
|---|---:|---:|---:|---:|
| Original merit | 83 | 45 | 36 | 28 |
| Merit plus log predicted support | 211 | 107 | 96 | 63 |
| Predicted support alone | 276 | 147 | 134 | 50 |

These releases use a different count, so the larger number of passing
views does not show a better fixed-budget policy. The combined score's
conservative trajectory difference is +0.000041 pp [-0.000022, +0.000094];
104 releases still fail or lack support for the original risk checks.
None of these rules is selected for deployment. The next method must model
support **and** the size of the conservative gain jointly, rather than use
support as a substitute for gain or relax the risk rule.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_opendd_fixed_joint_release.py freeze
.venv-pytorch/bin/python scripts/run_m3w_opendd_fixed_joint_release.py calibrate
.venv-pytorch/bin/python scripts/run_m3w_opendd_fixed_joint_release.py score
```

The [score freeze](score_freeze.json) was published in `0ff9ee218` before
calibration; the [calibrated thresholds](calibration.json) were published in
`2d1bf7a32` before held scoring. `fresh_run`: causal scores and both readouts.
`cached_verified`: matched proposals, support checkpoints and cohort.
`not_run`: new training, CREATE work or independent external evaluation.
See the [protocol](protocol.md) and [full aggregate readout](readout.json).
