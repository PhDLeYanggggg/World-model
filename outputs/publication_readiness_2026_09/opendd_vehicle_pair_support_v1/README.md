# OpenDD vehicles: two-edit joint ranking does not improve paired utility

The previous one-edit study found that putting easy support first yields
more qualifying replacements but gives up total gain. I tested whether a
pair could use support from one vehicle while preserving useful gain
from the other. I reused the site-excluded forecasters, factorized costs
and actor support models without refitting.

## Same query, two edits, same radii

Each recording supplies one eligible query selected from observed inputs.
It has at least three observed vehicles and allows two distinct edits
from the same forecast family. All three arms choose exactly two actors
at the same displacement radius. The independent and same-family arms
sum per-actor merits. The joint arm combines the gain and excess-risk
terms before scoring and uses the greater marginal support probability
as a lower bound on support of at least one actor. Since the marginals
are predictions, that construction does not certify actual risk.

The seven locations and three seeds give 126 held/calibration directions.
I froze choices and calibration thresholds at `cab820ea2` before held
scoring. All arms use the original 2% rules.

| Matched comparison | Independent | Same family | Joint |
| --- | ---: | ---: | ---: |
| Recording views | 8,874 | 8,874 | 8,874 |
| Edits | 17,748 | 17,748 | 17,748 |
| Missing easy denominator | 5,720 | 5,707 | 5,624 |
| All original risk checks pass | 2,281 | 2,334 | 2,371 |
| Also beat CV and OLS4 | 1,894 | 1,928 | 1,979 |

Joint adds 85 matched passes over independent. The difference is
positive in all seven location averages (+0.675 qualifying recordings
per view, 95% site bootstrap interval +0.341 to +1.095). The 8,874
views repeat 493 recordings over seeds and calibration choices. They
are not independent samples.

| Direct paired joint-minus-control contrast | Independent | Same family |
| --- | ---: | ---: |
| Conservative trajectory gain (pp) | -0.000738 [-0.001424, -0.000230] | -0.001598 [-0.003270, -0.000470] |
| Available-step trajectory gain (pp) | -0.000107 [-0.000331, +0.000094] | -0.000471 [-0.000941, -0.000153] |
| Conservative relative-motion gain (native units) | -0.00728 [-0.01022, -0.00465] | -0.01062 [-0.01639, -0.00574] |

Positive gain would favor joint. Trajectory gains are divided by the
same CV error mass. Relative motion is measured over all pairs on the
same query roster. The conservative results charge the actual
trajectory difference between the two arms on missing steps, rather
than subtracting two separately computed lower bounds. They do not
prove that the unknown-label errors are higher; they rule out a positive
conservative advantage in this experiment. Intervals use 3,000 paired
site resamples after averaging seeds and directions.

## The released policy still fails

| Held outcomes after independent-site calibration | Independent | Same family | Joint |
| --- | ---: | ---: | ---: |
| Released recordings (two edits each) | 56 | 56 | 61 |
| Released recordings without easy denominator | 39 | 39 | 43 |
| Released and passing all risk checks | 16 | 16 | 17 |
| Released and also beating CV and OLS4 | 15 | 15 | 13 |
| Active directions where every release passes both | 1 / 12 | 1 / 12 | 1 / 16 |

The frozen thresholds allow more joint releases, not more successful
ones. Release counts differ, so this table is separate from the
matched comparison. No arm is promoted; CV remains the fallback.

## The candidates still have substantial unused capacity

I enumerated all pairs of distinct actors and their legal candidate
paths in the same frozen query, without changing either edit radius.
Using future labels only for this diagnosis, **8,062 / 8,874** recording
views (90.85%) have at least one two-edit pair that beats both references
and satisfies the same risk rules. All found witnesses passed the
original scorer. The causal joint rule passes in 1,979 / 8,874 views
(22.30%). This separates the availability of a feasible combination
from the ability to recognize it using past observations. It does not
show that the hindsight pair is learnable from eight steps or that the
same fraction would hold on a new dataset.

The next useful comparison is to train a two-action feasibility and
gain score on source-excluded candidate pairs, rather than reconstruct
the pair score only from marginal gains and support. Keep the additive
and same-family controls, then freeze and calibrate before a new external
test. This is a recording-constrained extension of per-example selection,
not a new proof of the [multi-expert deferral surrogate](https://proceedings.mlr.press/v235/mao24d.html).

## Replay

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_vehicle_pair_support.py freeze
.venv-pytorch/bin/python scripts/run_m3w_vehicle_pair_support.py score
.venv-pytorch/bin/python scripts/diagnose_m3w_vehicle_pair_support.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_vehicle_pair_support.py tests/test_m3w_vehicle_pair_hindsight.py tests/test_m3w_vehicle_support_rank.py tests/test_m3w_vehicle_easy_factor.py tests/test_m3w_vehicle_matched.py tests/test_m3w_vehicle_residual.py
```

`fresh_run`: exact causal pair choices, site calibration, held scoring
and exhaustive matched hindsight diagnosis. `cached_verified`: all three sets of double-excluded weights
and the source cohorts. No fresh training or CREATE job. Nineteen
focused tests passed. Four local compute threads, no data workers,
hash-checked per-cell checkpoints and PID heartbeats. Songdo and
DroneCrowd confirmation were not scored. No manuscript, deployment,
Stage5C or SMC change.

[Protocol](protocol.md) | [Freeze](freeze.json) | [Readout](readout.json) |
[Matched hindsight capacity](hindsight.json).
