# OpenDD: choosing better swap queries is not enough

I changed which queries get a recipient swap, while holding the choice
of swap within every query fixed. The archived v194 gain head nominates
one swap per legal query. Each comparison takes 353 of those nominations
across the same held directions and seeds. At each direction both new
rules spend 95--100% of the sum of peak displacements of the archived
mask's nominations. Every forecast still has the original two edits and
the original displacement at each future step.

| Query rule | Swaps | Movement relative to archived | Queries with a qualifying option | Nominated swaps that qualify | Whole-roster risk / dual passes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen additive, no swap | 0 | -- | -- | -- | 4,379 / 2,975 |
| Archived query mask | 353 | 100.00% | 145 | 54 | 4,343 / 2,969 |
| Top predicted gain | 353 | 97.99% | 135 | 40 | 4,346 / 2,967 |
| Learned query opportunity | 353 | 97.71% | 163 | 68 | 4,362 / 2,984 |
| Hindsight allocation, same nominations | 353 | Constrained to 95--100% | -- | 190 | Not a release |

A qualifying swap improves on the frozen additive forecast and passes
the unchanged 2% recording checks and positive conservative gain against
both CV and OLS4. The query model changes 255 of the archived choices.
Its pooled qualifying count goes from 54 to 68, but the equal-location
rate change is **+3.21 pp [−0.33, +7.09]**. Coverage of possible swaps
rises by **+5.25 pp [−5.75, +17.32]** on the same location scale.
There are 493 distinct recordings at seven locations, 8,874 repeated
views and three training seeds. Intervals resample locations 3,000 times.

| Learned query rule minus | Conservative trajectory gain (pp), 95% interval | Conservative relative-motion gain (native units), 95% interval |
| --- | ---: | ---: |
| Archived query mask | −0.000152 [−0.000399, +0.000001] | −0.002547 [−0.004689, −0.001124] |
| Top predicted gain | −0.000180 [−0.000495, +0.000008] | −0.002638 [−0.005081, −0.001050] |
| Frozen additive, no swap | −0.000244 [−0.000445, −0.000104] | −0.001883 [−0.003281, −0.000907] |

The direction-averaged available ADE/FDE are 5.158126/15.174620 for
the query rule and 5.158111/15.174608 for no swap, in native coordinates.
Available p95 remains 12.483063 and worst observed signed easy
degradation remains 17.67%. The pooled whole-roster dual count rises by
nine over no swap, but the paired interval crosses zero. I did not
change the deployed policy.

## What the cap reveals

The earlier 2,624 opportunities are the queries with *some* qualifying
swap. The fixed gain nomination is qualifying in only 696 of them.
Under the same per-direction quota and movement range, the hindsight
allocator can capture 190. All 109 nonempty allocation problems solve to
optimality; 17 directions have zero quota. Thus neither better query
ranking alone nor counting all available actions gives the expected
forecast gain. I would next train the allocation score on signed gains
and harm of the actual nominated swap, including its effect on relative
motion, rather than on whether any good swap exists. This differs from
[decision-focused ranking](https://proceedings.mlr.press/v162/mandi22a.html):
the current query head has a binary training target, and the budget
solver is used after training. It is an offline diagnostic, not an
online policy.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_opendd_recipient_query_mask.py freeze
.venv-pytorch/bin/python scripts/run_m3w_opendd_recipient_query_mask.py score
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_recipient_query_mask.py
```

`fresh_run`: 21 native arm64 PyTorch fits and 6,300 updates (20.8 s),
then full held scoring (72.7 s). Four threads, no loader workers,
checkpoint every 50 updates and a PID heartbeat. The pre-score freeze
was published as `a79b6ad69`. `cached_verified`: OpenDD packets, aligned
past features, v194 nomination heads and query mask; replay reuses the
21 fits and 126 score receipts. The related suite passed 176 tests.
No external outcomes were opened and no CREATE jobs were started.

[Protocol](protocol.md) | [Freeze](freeze.json) | [Paired results](readout.json) |
[v195 paper](../evidence_manuscript_v195_recipient_query_mask/README.md)
