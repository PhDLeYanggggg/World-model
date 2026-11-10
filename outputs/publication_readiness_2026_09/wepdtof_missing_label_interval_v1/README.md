# WEPDTOF: two-sided sensitivity to missing positions

The frozen comparison reported a conservative lower score. I used the
same paired scalars to check what can also be ruled out under an
optimistic completion. No forecasts, labels or registration changed.

| Fixed joint minus | Observed contribution (pp) | Missing-position envelope (pp) | Empirical bootstrap outer range (pp) |
| --- | ---: | ---: | ---: |
| Independent | -0.01371 | [-0.04501, +0.01760] | [-0.11472, +0.09267] |
| Common family | +0.04475 | [-0.02057, +0.11006] | [-0.10661, +0.20048] |
| No pair penalty | 0.00000 | [0.00000, 0.00000] | [0.00000, 0.00000] |

The bounds do not resolve the sign against independent or common-family
selection. Against independent, four sources have a strictly negative
upper bound, three have a strictly positive lower bound, two tie, and
one is unresolved. The pooled bound still spans zero after dropping
any one of nine of the ten estimable sources. Equal scores against
the no-pair control here are not evidence of general equivalence.

## What the intervals mean

If the known contribution is `K` and the original conservative score
is `L`, the unknown contribution has magnitude at most `K - L` by
the reverse triangle inequality. The upper score is `2*K - L`. These
are valid bounds for this observed CV denominator; using a common
unknown path across seeds may make the joint interval tighter. The
bounds do not impute positions or divide by the unavailable full-grid
CV error.

Ten of 13 sources are estimable under the original matched rule. No
undefined contrast was replaced by zero. First average clips and seeds
within source, then resample sources. The outer range uses the 2.5th
percentile of the lower endpoint and the 97.5th percentile of the
upper endpoint under exact empirical resampling. It shows sampling
sensitivity, not a coverage guarantee. The original lower estimate,
3,000-draw interval and exact interval all reproduce from the archive.

The 2% risk checks and the original **0/45** dual passes remain the
decision evidence. The optimistic endpoint does not certify release.
The 15 clips, 13 sources and three seeds are the already exposed
one-shot cohort, not a new confirmation.

```sh
.venv-pytorch/bin/python scripts/audit_m3w_wepdtof_missing_label_interval.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_wepdtof_missing_label_interval.py tests/test_m3w_wepdtof_scalar_replay.py tests/test_m3w_wepdtof_frozen_capacity.py
```

Fifteen focused tests passed. [Aggregate receipt](receipt.json).
