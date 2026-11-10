# Learn the direction, keep the displacement fixed

I fit a smooth angular correction to the existing CV-to-OLS4 residual.
The neural head changes where the trajectory moves, but not how far
it moves from CV at any step. It sees the eight observed positions,
expressed in the CV/OLS4 disagreement frame, and the existing bridge
fraction. I compared the same initialized architecture and donor draws
under observed step error and conservative gain against two references.
The held location is excluded from fitting; causal references require
no learned teacher.

| Arm | Guarded dual passes / 606 | All-eligible dual passes / 606 | All-eligible risk passes |
| --- | ---: | ---: | ---: |
| Incumbent without exchange | 60 | 60 | 88 |
| Original exchange | 47 | 53 | 96 |
| Bounded acceleration | 60 | 64 | 95 |
| Decaying turn | 57 | 65 | 92 |
| Observed-suffix choice | 55 | 61 | 93 |
| Learned, observed error | 60 | 78 | 109 |
| Learned, dual-reference loss | 61 | 80 | 111 |

Each comparison preserves 780 original edits. The first mask changes
the incoming trajectory at the same 101 exchanges; the second uses
the prior ranker's fixed actions on all 296 legal views. Under the
second mask, the dual-reference model's equal-location pass-rate gain
against no exchange is +3.65 pp (95% interval [+0.67, +7.39]),
but its conservative utility difference is **-0.02883 pp**
[-0.05432, -0.00650]. The direct utility gain against the ordinary
learned loss is +0.00076 pp [-0.00002, +0.00224], not a reliable
objective advantage. Available endpoint mean error is 1.227128
versus 1.227168 without swaps, in native coordinates. This pooled
available-endpoint number is not a location-macro conservative result.

I trained 42 heads over seven held locations and three seeds:
16,800 optimizer updates in 20.8 seconds on native arm64. The
9.6-second replay matched with zero updates. The follow-up
[debit decomposition](../opendd_exchange_angular_debit_v1/README.md)
and [same-recipient test](../opendd_exchange_angular_same_recipients_v1/README.md)
separate the direction from the decision to exchange. No release,
external scoring, or change to the original 2% risk rule.

[Protocol](protocol.md) | [Readout](readout.json) | [Checkpoint receipts](head_freeze.json)

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_opendd_exchange_angular_head.py
.venv-pytorch/bin/python scripts/run_m3w_opendd_exchange_angular_head.py --replay
```

Four Torch threads and no data workers. The training command resumes
from the latest local 100-step checkpoint. The replay cannot update it.
