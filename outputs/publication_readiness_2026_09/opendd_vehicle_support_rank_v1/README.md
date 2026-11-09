# OpenDD vehicles: support ranking trades total gain for more feasible edits

I tested whether identifying the edited vehicle separately from choosing
its forecast can help when the easy-risk denominator is often undefined.
The vehicle forecasts, legal path bank, query, radius and one edit per
recording are unchanged. The original factorized choice and thresholds
are replayed. Two new rules prioritize recipient support before choosing
that recipient's best factorized-value path.

The learned support score uses eight observed positions, their motion
and three past-prefix CV backtests. It has no candidate-path or future
input. Its donor-only target is whether the actor has positive observable
CV error at or below the donor easy cutoff, so an edit could contribute an
easy reference. The non-trained comparison ranks by the worst prefix
CV error. Both new thresholds are set on a site excluded from the
forecaster, factorized head and support head for that direction. The
freeze is `2ce9cf5ba`, before held scoring.

## Same edit count, different recipients

| Result across repeated views | Archived value rank | Learned support rank | Prefix support rank |
| --- | ---: | ---: | ---: |
| Matched proposals | 9,000 | 9,000 | 9,000 |
| Easy denominator defined | 2,099 | 3,693 | 3,818 |
| All original risk checks pass | 1,473 | 1,942 | 2,030 |
| Also beat both CV and OLS4 | 1,114 | 1,647 | 1,348 |

The learned rank adds 533 risk-and-dual-utility passes over the archived
rule. Averaging within site, the count difference is **+4.23 per
site-direction view** (95% site bootstrap interval **+2.07 to +6.63**).
It improves that count in six of seven locations. But its conservative
gain over CV falls by **0.00380 pp** relative to the archived rule
(95% interval **-0.00596 to -0.00176 pp**). More small qualifying
edits did not translate into higher total utility.

The simpler prefix rule finds slightly more easy references but fewer
dual passes. In a secondary matched comparison, the learned rank beats
prefix by +2.37 dual passes per view (95% interval +0.008 to +4.91);
its conservative CV gain is +0.00172 pp higher (+0.00068 to +0.00271).
These descriptive intervals use 3,000 paired site draws, after averaging
seeds and calibration directions within each of seven sites. The 9,000
proposals repeat 500 recordings at three seeds and six calibration
choices; they are not 9,000 independent trials.

## Calibration did not produce a policy that passes on held sites

| Frozen release outcome | Archived value rank | Learned support rank | Prefix support rank |
| --- | ---: | ---: | ---: |
| Directions with qualifying calibration edits (of 126) | 17 | 42 | 4 |
| Held releases | 20 | 57 | 9 |
| Held releases without an easy denominator | 15 | 30 | 8 |
| Held releases passing all risk and both utility checks | 5 | 18 | 0 |
| Active held directions where every release passes | 0 / 7 | 3 / 19 | 0 / 3 |

For learned ranking, 30 releases have no easy denominator and nine
more have a defined denominator but fail the original risk checks.
Fourteen of its 18 passing releases come from rdb1. Release totals
are not count matched, and abstentions do not count as passes.
I retain CV as fallback.

**Paper implication.** Support can be learned well enough to improve
the number of feasible choices at fixed slots, but support alone is
not the optimization target. This one-edit study does not show a
joint-model gain. A useful next comparison is to combine support with
candidate-specific gain and harm in the same choice, rather than always
choosing the highest-support actor. Keep a matched utility control and
measure both total gain and feasible count. This differs from the
per-example routing objective in
[Regression with Multi-Expert Deferral](https://proceedings.mlr.press/v235/mao24d.html):
the original recording-level checks are applied after routing. It does
not inherit that paper's loss-consistency result.

## Reproduce

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_vehicle_support_rank.py freeze
.venv-pytorch/bin/python scripts/run_m3w_vehicle_support_rank.py score
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_vehicle_support_rank.py tests/test_m3w_vehicle_easy_factor.py tests/test_m3w_vehicle_matched.py tests/test_m3w_vehicle_residual.py
```

`fresh_run`: 63 native arm64 PyTorch fits, 12,600 updates and 126 held
readouts. Four compute threads, no data workers, checkpoints every
100 updates and PID heartbeats. Fourteen focused tests passed.
`cached_verified`: site cohorts, prior forecast and value checkpoint
hashes, and the archived factorized proposals. Its held counts replay
exactly. `not_run`: Songdo scoring and reserved DroneCrowd confirmation.
No deployment, manuscript, Stage5C or SMC change.

[Protocol](protocol.md) | [Freeze](freeze.json) | [Per-site/seed readout](readout.json).
