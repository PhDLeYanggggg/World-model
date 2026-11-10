# Where the one-exchange opportunity is lost

I replayed the frozen map-and-peer ranker and scored every legal
one-edit exchange, including recordings where the ranker stayed with
the incumbent. This kept the original 606 OpenDD development views,
7,198 legal exchanges, 780 realized edits, movement caps and 2% rule.

| Step | Views |
| --- | ---: |
| Incumbent dual passes without exchanging | 60 |
| In hindsight, incumbent fails but a legal exchange would pass | 121 |
| Of those: model does not act | 77 |
| Of those: model acts | 44 |
| Model chooses a passing exchange among those 44 | 5 |
| Incumbent passes destroyed by the model | 18 |
| Final model dual passes | 47 |

Among the 77 missed by the guard, 32 had no positive predicted gain
and 45 had no eligible transition score above the frozen margin. Of
the 39 wrong selections on rescuable views, 26 violated a measured
risk constraint, 11 lacked risk support and 2 passed risk but failed
to beat both utility references. Of 18 destroyed incumbent passes,
12 became measured risk failures, 2 lacked risk support and 4 lost
utility. Only two destroyed passes were unavoidable under the
forced-swap mask.

An oracle limited to the same 101 exchange views reaches 102/606
dual passes, versus 181/606 when it can also choose whether to
intervene. So ranking and admission both matter. At the fixed
mask, ranking by the same head's predicted gain gives 54 passes
and minimizing movement gives 45. Exact chance gives 51.21.
Gain ranking does not reliably beat chance (paired equal-location
rate difference -0.301 pp, 95% interval [-1.349, +0.365]); the raw
counts and equal-location mean weight recordings differently.

The dominant failure inside the mask is risk, not a shortage of
potential forecast gain. This motivates directly supervising risk
support and feasibility. The hindsight ceilings and alternative
rankings are development diagnostics, not changed release criteria.

[Protocol](protocol.md) | [Grouped readout](readout.json)

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/audit_m3w_opendd_exchange_attrition.py
```
