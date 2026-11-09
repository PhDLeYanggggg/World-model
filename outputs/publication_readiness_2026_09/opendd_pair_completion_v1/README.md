# OpenDD: learning the missing-step debit separately does not help

After finding that missing-label disagreement dominates several
matched pair contrasts, I tested whether learning that part of the
ranking score separately helps. Both refits use the same symmetric
pair network, five outputs, donor samples, initialization and
optimization budget. The direct control learns both conservative
gains plus an observed-gain auxiliary. The factored head learns the
common observed gain and a separate debit for each reference, then
reconstructs the two conservative ranking scores. Both keep the
original two risk targets. Future masks and errors are labels only.

The two-edit count, radius, query, candidate forecasts and source
roles are fixed. All choices and calibration thresholds froze at
`2166627d2` before the held readout. The 8,874 views per arm repeat
493 recordings across three seeds and six calibration directions.

| Fixed-budget result | Archived additive | Archived joint | Direct + auxiliary | Factored debit |
| --- | ---: | ---: | ---: | ---: |
| Easy denominator undefined | 3,730 | 4,247 | 4,157 | 4,146 |
| Pass original risk checks | 4,379 | 3,743 | 3,818 | 3,882 |
| Also beat CV and OLS4 | 2,975 | 2,952 | 3,035 | 2,991 |

| Factored minus comparator | Observed trajectory (pp; 95% CI) | Conservative trajectory (pp; 95% CI) | Conservative relative motion (native; 95% CI) |
| --- | ---: | ---: | ---: |
| Direct + auxiliary | -0.000091 [-0.000325, +0.000171] | -0.002534 [-0.004026, -0.001354] | -0.02073 [-0.02738, -0.01507] |
| Archived additive | +0.000053 [-0.000600, +0.000706] | -0.004719 [-0.007940, -0.002106] | -0.03773 [-0.05323, -0.02341] |
| Archived joint | -0.000195 [-0.000507, +0.000133] | -0.003279 [-0.005624, -0.001444] | -0.02491 [-0.03593, -0.01557] |

The primary count contrast is -0.349 dual passes per direction
[-1.333, +0.317], so it does not show a stable change in qualification.
The conservative trajectory and relative-motion contrasts are both
negative. These are direct paired bounds using the same missing-step
charge, not measured errors on unavailable labels. Intervals are
3,000 paired bootstrap resamples of seven location means.

Release is evaluated separately. The direct head releases 76 views
and qualifies 32; the factored head releases 108 and qualifies 47.
Only 6/23 and 3/25 active directions, respectively, qualify on every
released view. The release budgets differ; the larger qualifying
count is not a matched utility improvement or a deployment pass.

**Decision:** do not promote the factored ranking and do not tune it
on these held views. Retain CV fallback. This refit tests a specific
factorization, including a nonnegative CV-debit projection, not
whether observability is in principle learnable. The next decisive
evidence is the predeclared independent comparison, not another
architecture swap on the same seven exposed locations.

## Reproduce

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_opendd_pair_completion.py freeze
.venv-pytorch/bin/python scripts/run_m3w_opendd_pair_completion.py score
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_vehicle_pair_completion.py tests/test_m3w_vehicle_pair_learning.py tests/test_m3w_opendd_pair_disagreement.py tests/test_m3w_vehicle_pair_support.py
```

`fresh_run`: 126 fits, 25,200 updates and 126 held readouts.
Four native CPU threads, zero data workers, optimizer and sampler
state every 100 updates, per-cell freeze receipts and heartbeat.
Fourteen focused tests passed. `cached_verified`: donor forecasters,
cohorts and the two archived controls. No external outcome access,
independent confirmation, manuscript edit, deployment, Stage5C or SMC.

[Protocol](protocol.md) | [Freeze](freeze.json) | [Paired readout](readout.json).
