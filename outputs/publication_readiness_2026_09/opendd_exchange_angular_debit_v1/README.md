# Why more passing recordings do not imply greater utility

I decomposed the frozen head's paired utility against the unchanged
incumbent, without selecting any different exchanges. Each value
is an equal-location mean in percentage points of CV error mass.

| Mask | Observed gain | Missing-step debit | Conservative lower gain |
| --- | ---: | ---: | ---: |
| Guarded 101 exchanges | +0.000441 | 0.000642 | -0.000200 |
| All 296 legal exchanges | -0.014143 | 0.014688 | -0.028831 |

Under the all-eligible mask, the observed-gain 95% location interval
is [-0.043844, +0.007218], while the conservative lower interval is
[-0.054320, -0.006495]. Missing steps account for about half the
negative point estimate. They do not explain it alone. The original
and learned directions pay exactly the same debit against no
exchange, because the incoming and dropped row displacements do
not change in magnitude. Learning the direction improves the observed
contrast against the original exchange by +0.015126 pp
[+0.000855, +0.036154], but the direct conservative contrast crosses
zero. The three paired bounds cannot be subtracted to infer a
fourth: each pair has its own disagreement debit.

[Protocol](protocol.md) | [Grouped components and intervals](readout.json)

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/audit_m3w_opendd_exchange_angular_debit.py
```
