# OpenDD: stronger simple controls on the same vehicle histories

I tested whether the vehicle forecaster's gain over CV survives
comparison with simple motion alternatives. The four causal paths
are anchored median velocity, gradual velocity smoothing, decaying
acceleration and decaying turn. Each has the same peak correction cap
from CV as the vehicle model. The blend fits four nonnegative weights
on the six donor locations; any remaining weight stays on CV.
It uses the incumbent's same 51,200 training row draws per site and
seed, a convex optimizer and a fixed smoothed completion loss.
The 21 solutions were frozen at `65808162b` before held scoring.

| Forecast | Available ADE | Available FDE | Conservative gain vs CV, pp [95% CI] | Original risk passes / 1,500 |
| --- | ---: | ---: | ---: | ---: |
| CV | 5.161 | 15.183 | Reference | Undefined without edits |
| OLS4 | 6.124 | 16.645 | -27.68 [-29.44, -26.01] | 0 |
| Anchored median velocity | 6.071 | 16.536 | -26.62 [-28.43, -24.85] | 0 |
| Gradual velocity smoothing | 5.855 | 16.461 | -26.32 [-28.67, -23.81] | 0 |
| Decaying acceleration | 4.610 | 14.405 | -5.82 [-9.26, -2.40] | 6 |
| Decaying turn | 5.170 | 15.096 | -12.94 [-16.91, -9.07] | 0 |
| Fitted mixture | 4.990 | 14.934 | -1.06 [-2.05, -0.09] | 64 |
| Vehicle incumbent | 4.104 | 13.715 | +11.54 [+9.27, +13.71] | 14 |

The 141,783 histories come from 500 recordings and seven locations.
Errors are site macro means in native coordinates. Three seeds apply
to the fitted models; fixed baselines are repeated only for alignment.
Intervals use 3,000 paired location resamples after averaging the
seeds. The incumbent's archived gain against CV reproduced in every
site-seed cell. Its advantage survives this baseline set on the
development cohort, not necessarily on independent scenes.

The blend assigned mean weights of 0.269 to smoothing, 0.466 to
acceleration and 0.265 to CV; median and turn weights were effectively
zero. It improved the sampled donor objective by only 0.124% on
average. On held data, the blend lost 17.49 pp of observed gain against
the incumbent [-18.76, -16.10], and its direct conservative contrast
was -27.11 pp [-29.86, -24.24]. This direct paired bound uses the
actual difference between the two forecast paths on missing steps;
it is not the difference of their CV-relative bounds.

The blend passed the original recording risk checks more often
than the vehicle model, but neither passes them across the required
recordings. The higher count does not establish a release policy.
CV remains the strongest tested simple conservative reference, and
the current vehicle model remains a candidate, not the fallback.

## Reproduce

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_opendd_kinematic_blend.py fit
.venv-pytorch/bin/python scripts/run_m3w_opendd_kinematic_blend.py score
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_vehicle_kinematic_blend.py tests/test_m3w_vehicle_residual.py tests/test_m3w_vehicle_eqmotion.py
```

`fresh_run`: four deterministic controls, 21 convex fits and full
held scoring. `cached_verified`: cohorts and the vehicle incumbent.
Ten tests passed. Fit checkpoints and heartbeats are local. No external
outcomes, policy change, manuscript edit, Stage5C or SMC.

[Protocol](protocol.md) | [Freeze](freeze.json) | [Site/seed results](readout.json).
