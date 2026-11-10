# OpenDD: learning what the nominated swap does

I replaced the previous yes-or-no opportunity target with four values:
observed individual benefit, individual harm plus missing-step debit,
observed relative-motion benefit, and relative harm plus missing-pair
debit. Their differences give direct conservative effects against the
same frozen forecast. The two new allocations use the same trained
network: one ranks by individual value, the other adds half the
relative value. Both keep the existing recipient nominations, 353 swaps
and the previous 95--100% movement range in each direction.

| Allocation | Swaps | Spend vs original quota budget | Qualifying swaps | Recording risk / dual passes |
| --- | ---: | ---: | ---: | ---: |
| Frozen additive, no swaps | 0 | -- | -- | 4,379 / 2,975 |
| v195 opportunity | 353 | 97.71% | 68 | 4,362 / 2,984 |
| Predicted individual value | 353 | 97.12% | 47 | 4,349 / 2,979 |
| Individual plus relative value | 353 | 96.66% | 43 | 4,339 / 2,968 |

A qualifying swap both improves the incumbent and passes the original
2% recording rule and the CV/OLS4 comparisons. No allocation clears
the full-roster check: worst observed signed easy degradation remains
17.67%, and direction-averaged available p95 is unchanged at 12.483063
native units. Available ADE/FDE for the combined rule is
5.158129/15.174619 versus 5.158126/15.174620 for v195. The 8,874 views
repeat 493 recordings across directions and three training seeds; the
intervals below use 3,000 paired resamples of seven locations.

| Direct contrast | Conservative trajectory gain (pp), 95% interval | Observed relative contribution, 95% interval | Relative missing-label bounds |
| --- | ---: | ---: | ---: |
| Individual minus opportunity | −0.000159 [−0.000337, −0.000019] | +0.000199 [+0.000047, +0.000426] | [−0.002322, +0.002720] |
| Combined minus opportunity | −0.000163 [−0.000335, −0.000026] | +0.000201 [+0.000040, +0.000436] | [−0.002177, +0.002578] |
| Combined minus individual | −0.000049 [−0.000091, −0.000014] | +0.000001 [−0.000023, +0.000026] | [−0.000522, +0.000525] |

Relative units are native coordinates. The observed column sums the
observed effect on the requested grid without imputing unobserved
pairs. The last column varies their unknown outcomes at fixed observed
data; it is not a confidence interval. The combined rule improves the
observed relative contribution over v195 but does not establish the
full-grid sign. Adding the relative term to the individual score does
not improve the observed relative contribution reliably, and lowers
the conservative relative endpoint (95% interval [−0.000940,
−0.000207]). There is no reason to promote either allocation.

## Prediction versus ranking

Across all 8,814 legal nominated actions, the new head is better than a
zero-effect prediction for native absolute error: individual MAE is
0.0275 versus 0.0361, and relative MAE is 0.0349 versus 0.0770. But its
mean site-seed ROC AUC for a positive conservative individual effect is
0.556, below 0.585 for the archived local gain output. Its corresponding
relative AUC is 0.569. Repeated nominations are not independent samples.
An improvement in mean prediction error did not improve sign ranking.
Unlike a [joint forecasting model](https://proceedings.mlr.press/v205/luo23a.html),
this experiment changes when to use an existing forecast, not how the
joint trajectories are generated. For the next causal test I would
separate observed effect from future pair support and test sign ranking
before changing the allocator again. The deployment is unchanged.

## Reproduce

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_opendd_realized_swap_value.py freeze
.venv-pytorch/bin/python scripts/run_m3w_opendd_realized_swap_value.py score
.venv-pytorch/bin/python scripts/audit_m3w_realized_swap_prediction.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_recipient_realized_value.py
```

`fresh_run`: 21 native arm64 CPU fits, 6,300 updates (16.5 s), then held
scoring (72.3 s) and the nomination-value audit. Four Torch threads,
zero loader workers, 50-step checkpoints and a PID heartbeat. The
predictions were published as `d6eac872e` before scoring.
`cached_verified`: OpenDD inputs, v195 nomination and decision packets,
21 fit receipts and 126 score receipts. Replaying used no new optimizer
updates. The expanded regression suite passed 179 tests. No CREATE job
or reserved external outcomes, Stage5C or SMC.

[Protocol](protocol.md) | [Freeze](freeze.json) | [Paired results](readout.json) |
[Prediction audit](value_audit.json) |
[v196 draft](../evidence_manuscript_v196_realized_swap_value/README.md)
