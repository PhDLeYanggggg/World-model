# OpenDD: EqMotion under the same vehicle forecasting budget

I tested a compact, deterministic [EqMotion](https://arxiv.org/abs/2303.10876)
adaptation against the current vehicle forecast on the same OpenDD
cohort, rather than importing scores from its pedestrian benchmark.
The two variants share the pinned author core and a single fixed
output, with or without three nearest observed neighbors. Both use
the focal comoving frame, the incumbent CV displacement cap and the
same completion loss and donor sampling. The full cohort includes
141,783 vehicle histories in 25,596 nonoverlapping queries and 500
recordings. Each location is held out in turn, for all three seeds.
All 42 final checkpoints froze at `45ac61ecc` before scoring.

| Forecast | Available ADE | Available FDE | p95 ADE | Conservative gain vs CV (pp; 95% CI) | Original risk passes / 1,500 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CV | 5.161 | 15.183 | 12.497 | reference | undefined without edits |
| OLS4 | 6.124 | 16.645 | 14.579 | -27.68 [-29.44, -26.01] | 0 |
| Vehicle incumbent | 4.104 | 13.715 | 9.720 | +11.54 [+9.27, +13.71] | 14 |
| EqMotion ego only | 4.000 | 14.287 | 9.815 | +9.87 [+8.19, +11.61] | 17 |
| EqMotion + neighbors | 4.510 | 13.914 | 10.087 | +6.19 [+4.21, +8.45] | 5 |

Errors are native-coordinate site macro means. Both EqMotion variants
also beat OLS4 on conservative gain relative to the same observed CV
error mass (+18.55 and +15.64 pp, respectively). The original 2%
risk checks are not equivalent to the signed mean easy gain. Adding
neighbors raises the mean easy error by 58.38% against CV. The ego-only
arm lowers that error by 20.90% but passes the full checks in only
17 recording-seed views.

| Direct paired contrast | Observed gain (pp; 95% CI) | Conservative gain (pp; 95% CI) |
| --- | ---: | ---: |
| Ego only vs incumbent | +0.85 [+0.07, +1.62] | -7.90 [-9.55, -6.37] |
| Neighbors vs incumbent | -4.72 [-5.58, -3.86] | -13.76 [-15.88, -11.59] |
| Neighbors vs ego only | -5.57 [-6.61, -4.42] | -15.13 [-17.58, -12.56] |

The direct conservative contrasts charge the distance between the
two forecasts on missing steps. They are not differences of the
two CV-relative lower bounds and do not establish the actual error
on unavailable steps. Intervals use 3,000 paired site draws after
averaging the three seeds. All seven location means of the observed
neighbor effect are negative in this setup.

**Decision:** keep the current completion-trained path as the vehicle
candidate and CV as the deployment fallback. At the fixed budget,
this EqMotion neighbor representation does not add useful forecasting
signal, and the ego-only observed ADE advantage does not survive the
conservative comparison. This is not a reproduction of the
published multi-output metrics or evidence against interaction modeling
in general. No new threshold, held-site tuning or external release.

## Run

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_opendd_vehicle_eqmotion.py fit
.venv-pytorch/bin/python scripts/run_m3w_opendd_vehicle_eqmotion.py score
.venv-pytorch/bin/python scripts/plot_m3w_opendd_vehicle_eqmotion.py \
  data/stage_cvpr2027_experiments/opendd_vehicle_eqmotion_v1/vehicle_forecast_control.pdf
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_vehicle_eqmotion.py tests/test_m3w_vehicle_residual.py tests/test_m3w_opendd_comoving_eqmotion.py tests/test_m3w_opendd_vehicle_observed_geometry.py
```

`fresh_run`: 42 fits, 16,800 updates and 21 held location-seed
readouts. Native arm64 CPU, four compute threads, no data workers.
Fit saves model, optimizer and sampler state every 100 updates, and
score saves completed location-seed records. Repeat the same phase
to resume. Thirteen tests passed; a real EqMotion synthetic
interruption test reproduced the continuous fit's weights exactly.
`cached_verified`: cohorts, past-only geometry, pinned author core
and incumbent scores. No CREATE job, Songdo or DroneCrowd outcomes,
manuscript edit, deployment, Stage5C or SMC.

[Protocol](protocol.md) | [Freeze](freeze.json) | [Per-location readout](readout.json).
