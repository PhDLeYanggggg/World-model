# What Cancels The Comoving Model's Forecast Gain?

I decomposed the [frozen ego-only EqMotion trajectories](../opendd_comoving_eqmotion_v1/README.md)
relative to CV, without refitting any forecaster. Each residual is
split into components parallel and perpendicular to the last observed
displacement (first-to-last chord if the last displacement vanishes).
The guard retains a component only if its final sign agrees with the
corresponding component of OLS4 minus CV. If the direction is undefined,
the guard returns no correction. The equal-peak control keeps the full
neural direction, but rescales it to the guarded residual's peak on
each row, including zeros.

## Where The Gain Goes

These are 10,475 distinct rows, seven exposed OpenDD development sites,
three paired forecaster seeds. Easy and hard cuts use only the other six
sites. Slice effects are additive contributions in percentage points
of the *whole site's* observed CV error mass, not percent improvement
within the slice. The intervals average the seeds before 3,000 paired
site bootstrap draws.

| Slice, determined after prediction | Rows | Full residual lower contribution [95% CI], pp |
| --- | ---: | ---: |
| Easy | 2,928 | -2.823 [-3.958, -1.831] |
| Hard | 2,393 | +2.435 [+1.355, +3.763] |
| Other observed | 5,101 | +0.068 [-1.578, +1.852] |
| Wholly unobserved | 53 | -0.139 [-0.265, -0.036] |
| All rows | 10,475 | -0.460 [-2.718, +1.647] |

Available-step gain is +0.803 pp, before the total missing-step debit of
1.262 pp. Wholly unobserved rows account for only part of the debit.
The easy loss is already present with labels. Nor does high past CV/OLS4
disagreement identify the hard-slice advantage: the top donor-quartile
disagreement group contributes -0.438 pp [-0.951, -0.041].

## Does Selecting Components Help?

| Raw forecast | Conservative gain vs CV [95% CI], pp |
| --- | ---: |
| Full residual | -0.460 [-2.718, +1.647] |
| Longitudinal only | -0.583 [-2.259, +0.827] |
| Lateral only | -0.556 [-1.635, +0.740] |
| Direction guard | -0.675 [-2.266, +1.013] |
| Equal peak, unchanged direction | +0.392 [-1.125, +1.884] |

The direct paired guard-minus-equal-peak effect is -1.357 pp
[-1.947, -0.874]. This direct bound cannot be computed by subtracting
the separate lower bounds against CV. Reducing the correction and
changing its direction are different operations; the sign rule adds
no evidence of useful directional selection.

At the original 1,457 intervention recipients, times and radii,
the guard uses the original shape if its residual is zero (308 slots).
It raises joint passes from 90 for the donor-prefix control to 92/606,
but its direct effect is -0.0075 pp [-0.0462, +0.0512]. All arms leave
334 views without selected easy support. Keeping only longitudinal or
lateral corrections gives 84 and 83 joint passes, respectively.

An outcome-aware rowwise choice among CV, full, longitudinal, lateral,
and guarded forecasts reaches +11.945 pp [10.340, 13.531] versus CV.
This **unbudgeted** diagnostic optimizes only conservative accuracy,
not recording risk, and is not an achievable causal-policy score.
It motivated the separate
[CV-difficulty experiment](../opendd_hardness_gate_v1/README.md).

## Reproduction

```sh
OPENBLAS_NUM_THREADS=1 .venv-pytorch/bin/python scripts/run_m3w_opendd_component_guard.py
```

`fresh_run`: component paths, matched controls, slices and intervals.
`cached_verified`: existing cohort, forecast predictions and controller
outputs, including exact original and EqMotion replays. The repeat run
agrees with the immutable aggregate. No new forecaster fits, candidate
promotion, calibration or independent confirmation. Original 15/30
DroneCrowd roles and 2% rule unchanged; no Stage5C or SMC. The 8/12
task is native annotation steps in dataset-local coordinates.

[Registration](registration.json) | [Full site/seed results](analysis.json)
