# Learning Directions Against Two Motion Baselines

I tested the expanded shape bank without moving any interventions.
All arms retain the original 943 actions, the same rows and times,
and the same per-row peak displacement. The three candidates are
the frozen neural path, the OLS4 direction, and an analytic direction
chosen by backtesting within the observed prefix. An analytic
candidate is unavailable if reaching the neural radius would
amplify its displacement. When unavailable, the analytic controls
retain the neural path rather than changing the action count.

Two matched heads fit on double-excluded donor forecasts. They
learn how changing direction alters conservative lower gain and
three CV risk excesses relative to the original neural edit.
The CV-only head uses a duplicated CV target in place of the OLS4
target. The dual-reference head maximizes the smaller predicted
gain change across the two references. The guarded arm also
requires no predicted increase in risk; the joint arm can combine
the two role choices only when both are observed at the same
query. No future first or second edit is an input. All predictions
use the observed eight-step history and a twelve-step request.

| Policy | Joint passes out of 606 | Risk passes | Undefined |
| --- | ---: | ---: | ---: |
| Frozen neural direction | 138 | 172 | 295 |
| OLS4 direction | 111 | 167 | 295 |
| Observed-prefix analytic direction | 78 | 113 | 295 |
| Learned CV-only | 128 | 165 | 295 |
| Learned dual-reference | 130 | 166 | 295 |
| Guarded dual-reference | 137 | 170 | 295 |
| Co-observed joint guard | 137 | 170 | 295 |
| Outcome-aware fixed-row shape search | 226 | 249 | 295 |

The primary conservative contrast, dual versus CV-only, is
**+0.02306 pp [+0.00063, +0.06124]**. The improvement is concentrated:
omitting the largest contributing site gives +0.00442 pp
[-0.00070, +0.01084]. The two-sided site sign-flip sensitivity
gives `p=0.1875` under a symmetric-site-error assumption. Intervals
average three seeds within each site before 3,000 site resamples;
these 606 views do not constitute 606 independent recordings.

The dual head does **not** improve on frozen neural directions:
-0.01570 pp [-0.08997, +0.03571], with eight fewer joint passes.
The guarded arm changes 23 of 943 shapes, leaves one fewer joint
pass than neural and gains -0.01466 pp [-0.04572, +0.00226]
relative to it. Joint and independent guarding give identical
forecasts across the 52 co-observed pairs. Relative-path gain is
therefore also zero in this comparison, not evidence for a joint
improvement. The worst site-seed easy net degradation is 3.40% in
the neural, CV-only and dual-reference arms; that aggregate
diagnostic is not the full per-recording risk criterion.

Changing shape alone does not recover the 295 undefined views
because the edited rows are unchanged. The 226 outcome-aware
witnesses show some room to learn shape choice, not a causal result.
The next comparison should allow the two edit queries to change
on development data and score their realized budgets against
matched controls, rather than fit another ranker to these same
recipients. Keep both reference forecasts in the evaluation.

## Reproduce

```sh
.venv-pytorch/bin/python scripts/run_m3w_opendd_dual_direction.py
.venv-pytorch/bin/python scripts/run_m3w_opendd_dual_direction.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_opendd_dual_direction.py
.venv-pytorch/bin/python scripts/freeze_m3w_dronecrowd_calibration_roles.py verify
```

`fresh_run`: 42 local arm64 Torch fits, 16,800 updates, four
threads and no worker processes, with checkpoints and heartbeat
every 100 steps. `cached_verified`: original frozen trajectories
and schedules; zero-update checkpoint replay.
`not_run`: DroneCrowd independent confirmation. The 15/30 split
and original 2% rule are unchanged. No deployment or paper
submission; Stage5C and SMC remain off.
