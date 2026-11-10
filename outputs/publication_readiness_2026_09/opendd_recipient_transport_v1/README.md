# OpenDD: Better recipients exist, but the ranker rarely finds them

In Songdo, changing the edited vehicles mattered more than changing the
paths on the same vehicles. I tested that idea on the existing OpenDD
vehicle development set. Each comparison keeps the same query and two
edits. One edited vehicle is fixed; the other edit can be moved to a
new vehicle in its local heading frame, with exactly the same correction
norm at each prediction step. Only past motion and spacing enter the
decision. This tests a specific recipient change, not unconstrained
forecast generation.

## The two learned rules did not beat the incumbent

| Rule | Reassignments | Positive conservative gain among reassignments | Original risk pass | Risk and both reference gains |
| --- | ---: | ---: | ---: | ---: |
| Frozen additive | 0 | Not applicable | 4,379 | 2,975 |
| Mean donor loss | 353 | 161 (45.6%) | 4,345 | 2,970 |
| Worst-source weighting | 420 | 142 (33.8%) | 4,340 | 2,964 |

The matched comparisons cover 8,874 repeated views of the same 493
recordings at seven locations. Means average the repeated directions
and three seeds inside each location. Intervals are from 3,000 paired
location resamples; they are not intervals over 8,874 independent
samples. Undefined easy support rises from 3,730 to 3,736 and 3,738.

| Contrast | Available trajectory gain (pp) | Conservative trajectory gain (pp), 95% interval | Conservative relative gain (native units), 95% interval |
| --- | ---: | ---: | ---: |
| Mean minus frozen | -0.000114 | -0.000168 [-0.000305, -0.000047] | -0.001645 [-0.002913, -0.000784] |
| Worst-source minus frozen | -0.000122 | -0.000239 [-0.000410, -0.000081] | -0.002178 [-0.003856, -0.000904] |
| Worst-source minus mean | -0.000008 | -0.000102 [-0.000208, -0.000030] | -0.001597 [-0.002655, -0.000821] |

The available trajectory gain for each learned rule is also negative
against frozen, with a 95% interval below zero. Missing-label debits
are not the whole explanation. All seven location-level conservative
trajectory contrasts are negative for each learned rule against frozen.
The differences in the dual-pass rates are less certain; all three
paired intervals include zero. The weighting resembles the aim of
[group-robust training](https://arxiv.org/abs/1911.08731), but is a smooth
worst-donor loss relative to the zero-gain predictor, not that paper's
full algorithm or a guarantee on a new location.

## The same menu has usable choices in hindsight

Of 8,874 views, 4,809 offer a positive conservative gain over the
incumbent if all outcomes are known. The more relevant count is
**2,624**: a swap improves on the incumbent and still passes the original
2% checks and beats both CV and OLS4. It can rescue **1,336** views where
the incumbent fails those checks. The original scorer verifies the
witnesses, with no budget change. The mean head finds **58** strict
improvements and **27** rescues, and the worst-source head finds **47**
and **24**, respectively. These rescues do not offset losses in previously
passing views. The hindsight menu uses future labels only for diagnosis;
no such choices are available at inference or used for training here.

This narrows the problem to recognizing which swaps both help and keep
risk support, rather than having too few paths to choose from. The next
development comparison should learn eligibility and raw benefit separately
on this same menu, with the existing unchanged-risk scorer, instead of
reweighting the same normalized gain loss again. No release policy changes.
Athens, Swiss and other unopened outcomes were not accessed. The results
use native coordinates and annotation steps, not metric or seconds claims.

## Reproduce

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_opendd_recipient_transport.py freeze
.venv-pytorch/bin/python scripts/run_m3w_opendd_recipient_transport.py score
.venv-pytorch/bin/python scripts/audit_m3w_recipient_transport_ceiling.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_recipient_transport.py
```

`fresh_run`: 42 local arm64 Torch fits, 12,600 updates (18 seconds),
held scoring (35 seconds), and hindsight enumeration (27 seconds).
Four Torch threads, zero loader workers, optimizer/sampler checkpoints
every 50 steps, and PID progress records. Forecast receipts were pushed
as `35dcfa9c1` before held scoring. `cached_verified`: the existing
OpenDD cohorts, pair packets, observed geometry and original matched
incumbent. Replay reused all 21 fit receipts and 126 scoring and ceiling
receipts, reproduced the reports, and made no optimizer updates. The
related regression suite passed 168 tests. No CREATE job was needed.
No Stage5C or SMC.

[Protocol](protocol.md) | [Frozen decisions](freeze.json) |
[Paired readout](readout.json) | [Hindsight counts](ceiling.json)
