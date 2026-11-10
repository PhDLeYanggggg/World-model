# Songdo: the three seeds share the same unknown position

I kept the original 160 confirmation flights, 14 linked source
components, three model seeds and matched forecasts. The analysis
reads the frozen label masks and recorded paired scores, but not
numeric future positions. The original observed and lower points,
component means and bootstrap intervals replay for all three controls.

The old missing-label debit pairs each seed with itself. A common
target's mean error difference is unchanged if one arm's seed
predictions are permuted before summation. Taking the lowest distance
over the six matchings tightens the outer bound. I also evaluated
attainable shared points at the forecasts themselves, for a check on
how loose the new bound is under the same actor-step relaxation.

| Joint minus | Original sample interval | Shared-seed sample interval | 95% source interval for shared upper | Missing steps tightened |
|---|---:|---:|---:|---:|
| Independent | [-0.442183, -0.101883] | [-0.441925, -0.102142] | [-0.545733, +0.266337] | 7 |
| Additive | [-0.447575, -0.108034] | [-0.446600, -0.109009] | [-0.526328, +0.224776] | 11 |
| Same family | [+0.140873, +0.293031] | [+0.140873, +0.293031] | [+0.021073, +0.715733] | 0 |

Units are 0.001 percentage points on the original observed CV
denominator. The observed sample still favors the two unrestricted
controls for every missing-position completion allowed by the bound.
The 3,000 source resamples of their upper endpoints span zero; this
does not establish a loss on future source components. The advantage
over the restricted same-family control is unchanged and does not
establish a release advantage. No risk ratio is redefined.

The witness intervals are nested inside the shared-seed outer bounds.
For joint minus independent, the witness range is
[-0.441884, -0.102336] in the same units. These witnesses attain endpoints
within the explicit candidate choices, not necessarily global extrema,
and are not coherent trajectories or predictions of the missing data.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/audit_m3w_songdo_seed_shared.py
```

`fresh_run`: shared-seed geometry, witnesses and component statistics.
`cached_verified`: frozen paths, masks and original confirmation scores.
`not_run`: numeric future joins, training, new external cohorts or policy
changes. [Protocol](protocol.md) | [Readout](readout.json).
