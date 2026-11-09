# OpenDD: easy-support reranking does not improve the matched decision

After finding that many joint edits lost the easy reference denominator,
I added a frozen, past-only actor support score to the original pair
ranking. The prediction bank, fixed query, two edits and displacement
radius did not change. A negative control applied the same penalty to
support scores shuffled within each query. No model was retrained. All
choices and calibration thresholds froze at `0c418204f` before held scoring.

| Same-budget result | Additive | Additive + support | Additive + shuffled support | Same family | Family + support | Joint | Joint + support |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Views | 8,874 | 8,874 | 8,874 | 8,874 | 8,874 | 8,874 | 8,874 |
| Easy denominator undefined | 3,730 | 3,755 | 3,719 | 4,210 | 4,246 | 4,247 | 4,289 |
| Pass all original risk checks | 4,379 | 4,269 | 4,370 | 3,835 | 3,718 | 3,743 | 3,624 |
| Also beat CV and OLS4 | 2,975 | 3,011 | 2,998 | 3,012 | 3,018 | 2,952 | 2,956 |

Additive plus support found 36 more dual passes than additive, but
the paired count interval spans zero (+0.286 per direction,
[-0.230, +0.778]). Easy support did not increase, and learned support
did not establish a count advantage over the shuffled control.

| Direct paired contrast | Observed-step trajectory gain (pp; 95% site CI) | Conservative trajectory gain (pp; 95% site CI) |
| --- | ---: | ---: |
| Additive + support vs additive | +0.000344 [-0.000022, +0.000877] | -0.000690 [-0.001091, -0.000412] |
| Additive + support vs additive + shuffled | +0.000133 [-0.000180, +0.000483] | -0.001171 [-0.001734, -0.000694] |
| Family + support vs same family | +0.000318 [+0.000076, +0.000620] | -0.000523 [-0.000842, -0.000266] |
| Joint + support vs joint | +0.000305 [+0.000076, +0.000591] | -0.000552 [-0.000897, -0.000254] |

The positive observed-only family and joint contrasts do not survive
the conservative missing-step check. All four learned-support contrasts
also lose conservative relative-motion utility. The direct paired
missing-step debits depend on both predictions, so the contrast
against shuffled support cannot be obtained by subtracting two
separately computed lower bounds. The 8,874 views repeat 493 recordings
across 3 seeds and 6 calibration directions. Intervals use 3,000 paired
resamples of seven location means.

Release is not a matched-budget comparison. Joint plus support released
126 views versus 66, with 55 versus 29 dual passes. All releases qualified
in only 2/19 active directions versus 4/16 for the original joint rule.
Neither passes deployment, and CV remains the fallback.

The result rules out this particular penalty as a fix, not every way of
modeling support. It does not justify tuning its weight on these held
views. The priority is independent evaluation of the already frozen
methods; any further support model should distinguish future annotation
availability from positive easy CV mass and predict both on donor data.

## Reproduce

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_opendd_support_rerank.py freeze
.venv-pytorch/bin/python scripts/run_m3w_opendd_support_rerank.py score
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_opendd_support_rerank.py tests/test_m3w_opendd_pair_disagreement.py tests/test_m3w_vehicle_pair_learning.py tests/test_m3w_vehicle_support_rank.py tests/test_m3w_vehicle_pair_support.py
```

`fresh_run`: candidate reranking, calibration and scoring across
126 directions; 16 tests passed. `cached_verified`: all fitted weights,
cohorts and three original control choices and counts. Four native
CPU threads and no data workers; per-cell checkpoints and PID heartbeat
during freeze. No new HPC job, independent external outcome access,
manuscript changes, deployment, Stage5C or SMC.

[Protocol](protocol.md) | [Freeze](freeze.json) | [Lightweight metrics](readout.json).
