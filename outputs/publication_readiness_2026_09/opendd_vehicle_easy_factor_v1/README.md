# OpenDD vehicles: separating easy support did not fix release

The previous vehicle policy made no edits because its predicted easy-risk
excess was positive for every candidate. I tested whether predicting the
existence of an easy reference separately from its conditional harm and
reference mass would produce better selections.

## Comparison

Both arms use the same vehicle forecasts, same fixed query and same edit
radius, with exactly one proposed edit per eligible recording. This is a
ranking experiment, not a test of two-agent joint selection. For each outer
site and seed, each of the other six sites serves once as calibration. The
forecast and risk model exclude both sites; the candidate forecast used to
build each risk-training batch additionally excludes its own donor site.
The release threshold sees only the calibration site's outcomes for its
direction. Choices and thresholds were committed at `8f7d31eec` before
outer scoring. The observed-history length, forecast horizon and 2% rules
are unchanged. These are seven OpenDD development locations, not seven
independent final tests.

| Outcome | Direct head | Factorized head |
| --- | ---: | ---: |
| Matched one-edit proposals | 9,000 | 9,000 |
| Matched proposals passing all original risk checks | 1,429 | 1,473 |
| Matched proposals also beating both CV and OLS4 | 1,082 | 1,114 |
| Matched proposals without an easy-risk denominator | 6,915 | 6,901 |
| Directions with a nonempty qualifying calibration prefix (of 126) | 16 | 17 |
| Released on held sites | 24 | 20 |
| Released without an easy-risk denominator | 11 | 15 |
| Released and passing all original risk checks | 8 | 5 |
| Released and also beating both CV and OLS4 | 7 | 5 |
| Active held directions with every release passing (out of 5 / 7 active) | 0 | 0 |

The 9,000 entries per arm repeat 500 recordings at three seeds and six
calibration choices. After averaging repeated directions and seeds inside
each site, the paired factorized-minus-direct gain is **+0.00056 pp**
against CV (3,000 site bootstrap draws; 95% interval **-0.00019 to +0.00149
pp**). The paired difference in risk-and-dual-utility pass count is +0.254
per direction, averaged across sites (95% interval -0.135 to +0.706).
Neither matched contrast establishes an improvement. Releases are not
matched across heads, so their gains do not isolate an effect of
factorization.

## Why the gate still fails

The factorized head released 20 edits across 19 distinct held recordings.
Its 15 failed release instances had no easy reference mass among the
edited rows, rather than an estimable easy ratio above 2%. All five
estimable releases passed the three risk checks and improved on both
references. The direct head had 16 failures: 11 missing easy denominators
and five estimable releases over both all and easy limits. The complete
policy does not pass. In particular, neither abstentions nor undefined
ratios count as risk passes.

The same training-time support estimate that enters the expected easy
ratio did not identify support reliably at release time. The next
development test should compare a dedicated causal support ranking
against the direct score at exactly the same slots, then test site-held
release separately. It should not treat a change in easy risk definition
as an improvement. Our score-prefix rule is empirical, unlike the
finite-sample testing setup in [Learn then Test](https://arxiv.org/abs/2110.01052).
No deployment or generalization claim follows. The frozen external
evaluation and the manuscript are unchanged.

## Replay

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_vehicle_easy_factor.py freeze
.venv-pytorch/bin/python scripts/run_m3w_vehicle_easy_factor.py score
.venv-pytorch/bin/python scripts/diagnose_m3w_vehicle_easy_factor.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_vehicle_easy_factor.py tests/test_m3w_vehicle_matched.py tests/test_m3w_vehicle_residual.py
```

`fresh_run`: 105 four-site producer fits and 126 risk fits, 79,800
optimizer updates, a frozen readout and failure attribution. Local
arm64 PyTorch, four compute threads, no data workers, checkpoints every
100 steps and training heartbeat. Eleven focused tests passed; repeating
`score` reproduced the same readout SHA-256. `cached_verified`: OpenDD
cohorts and the 63 two-site-excluded forecast checkpoints used for held
inference. `not_run`: Songdo scoring or DroneCrowd confirmation. No
Stage5C, SMC, deployment change or physical safety claim.

[Protocol](protocol.md) | [Freeze](freeze.json) | [Readout](readout.json) |
[Failure counts by site](diagnosis.json).
