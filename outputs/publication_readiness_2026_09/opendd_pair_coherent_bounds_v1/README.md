# When shared missing positions change a paired comparison

I tested whether the OpenDD relative-motion comparison was inconclusive
because its missing-label bound treated each pair separately. One missing
actor has one position at each step, even when it contributes to several
pairs. The [assignment bound](method.md) uses that shared position.
A separate check constructs explicit completions with opposite effects.

| Fixed interventions | Direct supervision minus | Changed views | Bound crosses zero | Both signs exhibited |
| ---: | --- | ---: | ---: | ---: |
| 526 | Anchor difference | 163 | 83 | 82 |
| 526 | Pointwise | 124 | 53 | 52 |
| 526 | Incumbent, after calibration | 0 | 0 | 0 |
| 1,368 | Anchor difference | 479 | 258 | 256 |
| 1,368 | Pointwise | 355 | 184 | 181 |
| 1,368 | Incumbent, after calibration | 5 | 3 | 3 |

The assignment did not tighten any of these comparisons. In the first
row, 3,064 of 3,131 informative missing-actor steps had one observed
neighbor with a changed distance anchor, 12 were near common translations,
and the other 55 also retained an optimal original matching. The new
mean bounds for this row are [-0.000247, +0.000216] in native coordinate
units. The bootstrap intervals for the lower and upper endpoints are
[-0.000463, -0.000059] and [+0.000038, +0.000440], respectively.
These are separate endpoint uncertainties, not a single confidence interval.

The key result is that the ambiguity is attainable for most individual
changed queries whose intervals cross zero. It is not just a loose
triangle bound. Different views can reuse labels, so the exhibited
completions cannot be combined into a single full-dataset trajectory or
taken as proof of physical plausibility. The original risk rule and the
choice to retain the incumbent are unchanged.

This is a fresh audit of already-scored development proposals, with
493 recordings, 8,874 repeated views and 126 fixed directions. It did
not fit a model or access a new outcome. The [readout](readout.json)
contains the endpoint estimates, site means, and intervals from 3,000
bootstrap draws over seven locations. The run checks the
archived estimates before computing the geometry in double precision.

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/audit_m3w_coherent_pair_bounds.py
```
