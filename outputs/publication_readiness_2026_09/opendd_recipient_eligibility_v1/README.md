# OpenDD: the switch mask misses most useful recipient changes

I kept the original two edits per query, the same forecast paths and
the same stepwise movement budget. The model learns two quantities:
the expected gain over the incumbent, and the probability that a swap
passes the original 2% recording checks and beats both CV and OLS4.
For the primary comparison I use both outputs from the same model and
the same 353 queries chosen by the archived mean-loss ranker. Only the
ranking changes. The guarded version may skip a swap; it is not an
equal-count control.

| Rule | Swaps | Swaps passing risk and both references | Swaps also improving incumbent | Whole-roster dual passes |
| --- | ---: | ---: | ---: | ---: |
| Frozen additive incumbent | 0 | -- | -- | 2,975 |
| Archived mean-loss ranker | 353 | 105 | 58 | 2,970 |
| New network, gain output only | 353 | 104 | 54 | 2,969 |
| Same network, prioritize eligibility | 353 | 91 | 20 | 2,956 |
| Eligibility plus positive-gain guard | 4 | 4 | 0 | 2,979 |

The eligibility score changes the selected swap in **221 of 353**
queries. Against gain-only, its direct conservative trajectory gain is
**-0.000135 pp** (95% location interval **[-0.000173, -0.000092]**),
and its conservative relative-motion gain is **-0.001367** native units
**[-0.001853, -0.000835]**. On available trajectory steps the gain
interval is **[-0.000031, +0.000117] pp**. The conservative and
available-step readouts therefore answer different questions. The
whole-roster dual-pass rate interval also includes zero. The worst
observed signed easy degradation remains **17.67%** in all arms, so
none can pass the original full-roster risk requirements. The guard's
four swaps improve the pass count, but none improves the incumbent's
conservative trajectory error. The score cutoff is not a guarantee
of risk control of the kind studied in
[Conformal Risk Control](https://openreview.net/pdf/482dcb57a6da0d60fce0a3b533b80bcbe99b03f6.pdf).

The opportunity count reproduces the earlier exact audit: **2,624**
of 8,874 repeated views have a strictly better swap that also passes
the risk and two-reference checks. Only **145 (5.53%)** of those views
are in the archived 353-query mask. Even a perfect ranker on that mask
cannot recover the other 2,479. There are 493 distinct recordings and
seven locations, not 8,874 independent samples. I average the three
training seeds inside locations and resample locations 3,000 times.

This separates two errors: deciding whether a query has a worthwhile
swap, and choosing a swap within that query. Neither adding an
eligibility score nor choosing a different cutoff on these same
353 queries addresses the first. The next development test should learn
query-level opportunity and compare new query masks at the same count
and displacement before reranking. The OpenDD results are development
evidence; the frozen independent results and deployment stay unchanged.

## Reproduce

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_opendd_recipient_eligibility.py freeze
.venv-pytorch/bin/python scripts/run_m3w_opendd_recipient_eligibility.py score
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_recipient_eligibility.py
```

`fresh_run`: 21 native arm64 CPU PyTorch fits and 6,300 updates, with
four threads, no data workers, 50-step checkpoints and PID heartbeats.
The frozen fit took 19.3 seconds; held scoring took 86.8 seconds. The
first smoke fit exposed a path typo during receipt validation, before
any held scoring. The corrected predictions were published at `8acf58cfa`
before scoring. `cached_verified`: previous OpenDD geometry and frozen
forecast packets, the 353-query mask and its outcome audit. The later
replay reused checkpoint and scoring receipts without optimizer steps.
The expanded regression suite passed 172 tests. No reserved external
outcomes were opened. No Stage5C, SMC, or CREATE submission.

[Protocol](protocol.md) | [Frozen decisions](freeze.json) |
[Paired results and diagnostics](readout.json) |
[v194 paper build](../evidence_manuscript_v194_recipient_eligibility/README.md)
