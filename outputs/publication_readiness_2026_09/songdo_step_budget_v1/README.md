# Songdo: equal displacement at each forecast step

The frozen Songdo comparison gave every proposal two edits and the
same maximum departure from CV. It did not require the same departure
at every step. I kept the recipients and edit directions fixed and
repeated the comparison under a common time profile. The profile uses
the smallest of the four departures at each step. A separate control
shrinks all four original paths uniformly to the new peak.

All 480 flight-seed views support the stricter comparison. On average
across sources it retains 94.8% of the original peak and 87.3% of the
sum of the original displacements. This sum measures departure from
CV, not physical travel distance.

| Budget | Joint minus independent observed gain (pp) | Conservative gain (pp) | 95% source interval for conservative gain | Risk and both utility passes, independent / joint |
| --- | ---: | ---: | ---: | ---: |
| Original matched peak | -0.000272 | -0.000442 | [-0.001499, +0.000237] | 169 / 53 |
| Same smaller peak, original profiles | -0.000270 | -0.000426 | [-0.001397, +0.000189] | 172 / 54 |
| Same departure at every step | -0.000259 | -0.000423 | [-0.001317, +0.000122] | 172 / 57 |

Matching the time profile rather than only shrinking the peak changes
the lower endpoint by +0.00000345 pp [-0.00006658, +0.00007824].
The same-sample missing-label upper endpoint after step matching is
-0.0000962 pp, but its source interval crosses zero. The paired
relative-motion lower contrast is -0.000279 native units
[-0.000633, +0.000026]. The stricter control does not resolve a joint
advantage or explain the original negative point estimate away.

At the matched profile, available ADE is 1.373117 / 1.373124 and
available FDE is 3.416738 / 3.416760 for independent / joint.
The risk and utility counts describe matched proposals, not released
flights. A stepwise adjustment is not a new dynamics model.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_songdo_step_budget.py freeze
.venv-pytorch/bin/python scripts/run_m3w_songdo_step_budget.py score
```

`fresh_run`: transformed and scored the existing 160 flights,
including 3,000 paired resamples over 14 source components.
`cached_verified`: original directions, recipients, CV/OLS4 references,
labels and risk cutoffs. The original contrasts and risk counts
reproduce. New transformations were pushed at `f6b9d07ab` before
the retrospective score. `not_run`: new training, calibration,
release decisions or external confirmation. No CREATE work,
Stage5C, SMC, or submission.

[Protocol](protocol.md) | [Forecast freeze](freeze.json) |
[Readout](readout.json).
