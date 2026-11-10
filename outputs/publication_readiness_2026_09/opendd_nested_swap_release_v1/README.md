# OpenDD: disjoint calibration exposes two failure modes

I tested whether the support-aware ranking gain is useful when the
policy may abstain. Each of the 63 fits uses five OpenDD locations.
Both other locations are excluded from its fitting and from the
forecast producers used to generate its training candidates. A donor's
producer also excludes that donor. Each fit then gives two directions:
calibrate on one excluded location, and score on the other.

The same head supplies a sign score and a signed magnitude score. The
sign arm can nominate any legal one-recipient swap; the magnitude arm
requires positive predicted net value. Neither changes the query, two
edit count or stepwise edit displacement in a nominated forecast.
Calibration retains the highest tied-score prefix for which every
recording passes the original 2% checks, beats CV and OLS4, and improves
the frozen additive incumbent. The archived additive release does not
require that last improvement over itself. The release arms have
different counts. On all other recordings the forecast is CV.

| Rule | Eligible calibration views | Calibration directions with a threshold | Held releases | Risk passes | Risk and both references | Also improves incumbent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Archived additive | 8,874 | 25 / 126 | 65 | 32 | 19 | Not applicable |
| Sign | 8,814 | 20 / 126 | 32 | 12 | 8 | 6 |
| Magnitude | 8 | 1 / 126 | 0 | No release | No release | No release |

Magnitude abstains before calibration on all but eight candidate views.
One calibration direction retains two views, but its held counterpart
releases none. This is a CV fallback, not zero measured risk. The sign
arm has 606 strict candidates in hindsight on calibration, but only 22
survive the tied-prefix rule. Of 32 held releases, 16 have an undefined
selected-easy denominator; another four fail risk despite an estimable
selected-easy denominator. The largest signed degradation across
observed easy rows is only 0.432%, so that aggregate alone would have
missed the failure. It is distinct from the two selected positive-harm
ratios in the original check.

| Forecast | Available ADE / FDE | Worst recording available p95 | Worst signed easy degradation |
| --- | ---: | ---: | ---: |
| CV | 5.160345 / 15.176156 | 15.451091 | 0.000% |
| Archived additive release | 5.160334 / 15.176150 | 15.451091 | 0.230% |
| Sign release | 5.160341 / 15.176150 | 15.451091 | 0.432% |
| Magnitude release | 5.160345 / 15.176156 | 15.451091 | 0.000% by fallback |

ADE/FDE are direction means on available annotations in native units;
the recording tail is a worst case over the roster. The sign arm's
conservative gain versus CV is **+0.000012 pp** (95% paired location
interval **[−0.000001, +0.000028]**). The 126 directions repeat 493
recordings over three seeds and six calibration roles. We average
those repeats before 3,000 location resamples. Gains versus OLS4
cannot be credited to a rule that never releases: the magnitude arm's
7.439 pp against OLS4 are exactly the CV fallback's gain.

This is an empirical frozen calibration check, not the population
guarantee of [Learn then Test](https://arxiv.org/abs/2110.01052).
The donor anchor uses a deterministic causal proposal and differs
from the additive anchor at calibration and held evaluation. The
experiment shows two distinct problems under the tested rule: the
magnitude score rarely nominates, and the sign score transports poorly
to a different location. The existing policy stays unchanged. I would
address selected-easy support directly before trying another value
weight; another ranking-only gain would not fill the independent
validation gap.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_opendd_nested_swap_release.py freeze
.venv-pytorch/bin/python scripts/run_m3w_opendd_nested_swap_release.py calibrate
.venv-pytorch/bin/python scripts/run_m3w_opendd_nested_swap_release.py score
.venv-pytorch/bin/python scripts/audit_m3w_nested_swap_release_tails.py
```

`fresh_run`: 63 native arm64 PyTorch fits, 18,900 updates (88.4 s),
calibration (32.9 s), held scoring (43.4 s), and tail audit. Four Torch
threads, zero data workers, 100-step optimizer and sampler checkpoints,
and PID progress. Predictions were published at `32e59d50a`, thresholds
at `54af59ec3`, each before the next phase. `cached_verified`: 105
triple-excluded forecast checkpoints, shared-frame past features,
archived additive proposals and thresholds. Cached fit and score replay
reproduced the release counts. 185 related tests passed. No new CREATE
jobs, external outcomes, Stage5C, SMC, or submission.

[Protocol](protocol.md) | [Calibration freeze](calibration_freeze.json) |
[Held results](readout.json) | [Error tails](tail_audit.json) |
[v198 draft](../evidence_manuscript_v198_nested_swap_release/README.md)
