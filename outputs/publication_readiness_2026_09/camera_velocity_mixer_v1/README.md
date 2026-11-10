# Velocity-only forecasts on camera histories

I trained nine small velocity mixers on the seven OpenDD donor cohorts
(3 seeds, 800 updates each). A mixer blends five observed slopes and
produces a linear path anchored at the last detection. The experiment
tests whether the starting forecast can improve before joint selection.

| Forecast | Known ADE (px) | Conservative gain vs OLS4 (pp) | Original easy positive-harm upper |
| --- | ---: | ---: | ---: |
| CV | 19.65 | -26.38 | undefined |
| OLS4 | 14.61 | 0.00 | 27.30% |
| Median4 | 13.95 | +3.03 | 22.89% |
| Median8 | 13.14 | +3.72 | 41.11% |
| Noise-trained spline (prior) | 13.48 to 13.73 | +2.49 to +4.36 | 29.67% to 33.84% |
| Clean mixer | 15.60 to 16.34 | -9.70 to -6.27 | 11.54% to 13.92% |
| Noise mixer | 13.63 to 13.86 | +2.18 to +2.92 | 24.53% to 29.88% |
| Noise + easy penalty | 13.53 to 13.98 | +1.80 to +3.72 | 16.33% to 21.50% |

Synthetic history noise helps the mixer in all three matched seeds.
The easy penalty lowers the measured harm in each seed, but no arm
passes the original 2% check. The simple median8 reference has the best
observed ADE and FDE (22.07 px), and is not a safe replacement. These
are 58 histories (57 complete futures) from the same previously exposed
camera, not independent external evidence. CV fallback stays unchanged.

The batch loss also exposes a normalization defect. Among 10,422 donor
rows, 450 have near-stationary prefixes and 186 of those have nonzero
future displacement. Dividing by the `1e-8` minimum inflates some donor
losses beyond 100,000. A convex mix of zero slopes cannot predict a start
from rest; that modeling limit is separate from the scale defect. Fit
a nonzero floor from donor past motion only and repeat the matched runs.

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/run_m3w_camera_velocity_mixer.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_camera_velocity_bank.py tests/test_m3w_wepdtof_frozen_capacity.py
```

Native arm64, four CPU threads, no workers. Checkpoints and heartbeat are
local; rerunning resumes from the last completed 100-step block.
Nine focused tests passed. [Registered setup](protocol.md),
[training and forecast freeze](freeze.json), [pilot readout](readout.json).
No non-pilot external targets, revised risk thresholds or paper edits.
