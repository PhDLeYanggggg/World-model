# DroneCrowd: Calibrating Risk at the Recording Level

The [query-level allocation](../dronecrowd_ols_relative_gain_v1/README.md)
could not find a viable replacement, even though OLS4 itself passed
the observed two-ratio check in 4,199 queries. Here I stop requiring
every query to meet its own predicted risk budget. The final score still
uses both original CV-referenced 2% limits and nonzero easy support on
each complete recording.

## Comparison

I reuse the three frozen CV-target heads. Starting at OLS4, each
query has exactly the previously frozen independent replacement count.
All four arms use the same candidate paths. A common price of 0, 1, 4
or 16 reorders the estimated gain by the change in predicted all and
easy harm excess relative to retaining OLS4. Price zero reproduces
the original paths exactly. These arms may change actors or peaks;
this is an equal-count diagnostic, not the exact two-edit experiment.
The allocation has no explicit interaction or collision term.

For each frozen arm, I sum predicted harm and reference mass across
the recording before taking their ratio. This includes default OLS4
paths when they differ from CV. I compare three ways to predict risk:
the raw model, an additive correction equal to the median of source
medians, and an additive upper envelope of observed-minus-predicted
residuals. Each held source is excluded from its corrections, including
all recordings and all three seeds. Missing observed support is treated
as a failure for fitting the offset, not as an observed risk value.
Each rule chooses the positive predicted OLS4 gain leader among arms
passing both predicted limits, or abstains. Abstentions are not counted
as passes. Observed evaluation requires the original risk checks and
conservative positive utility over both CV and OLS4.

The 40 recordings (1,757,584 rows) come from the same 15 previously
opened calibration source groups. The results are leave-one-source-out
development estimates, not independent confirmation. Utility intervals
are paired over 3,000 bootstrap draws of source groups; recording-seed
views are not independent. I do not treat these offsets as
[Conformal Risk Control](https://arxiv.org/abs/2208.02814) or
[Learn then Test](https://arxiv.org/abs/2110.01052): the loss here need
not be monotone across prices, and this experiment provides no new
finite-sample risk guarantee. Coordinates are native image units,
with eight observed and twelve forecast annotation steps.

## Results

Each arm replaces exactly 3,824,730 OLS4 paths across the 120
recording-seed views, with counts matched in every query.

| Arm | Original risk pass | Both accuracy floors and risk | Conservative gain vs independent (pp) | vs OLS4 (pp) |
|---|---:|---:|---:|---:|
| Independent | 24 | 24 | 0.000 | +0.144 |
| Price 1 | 25 | 13 | -0.090 [-0.410, +0.255] | +0.055 |
| Price 4 | 28 | 0 | -0.911 [-1.525, -0.299] | -0.764 |
| Price 16 | 29 | 0 | -1.010 [-1.641, -0.376] | -0.861 |

The highest price adds five risk passes without losing a risk pass,
but loses all 24 joint passes of the independent control. Price 1
adds one joint pass and loses twelve. The union has 25 joint passes,
only one more than the independent arm, with outcomes known. From
price 0 to 16, mean available ADE/FDE worsens from 1.4241/2.5527 to
1.4626/2.6063 px, while the worst recording p95 available ADE improves
from 6.7050 to 6.6392 px. Lower tail error alone is not a joint pass.

| Held-source correction | Selected views | Actual risk pass | Both accuracy floors and risk |
|---|---:|---:|---:|
| Raw | 0 | 0 | 0 |
| Source median | 13 across 3 sources | 12 | 4 |
| Source upper envelope | 0 | 0 | 0 |

The median correction recovers four of the 25 views with an actual
passing arm. It excludes all other 21 through the estimated risk
constraints, not the predicted gain sign or rank. The upper envelope
excludes all 25. The point correction also selects one view that fails
risk and eight that pass risk but not both accuracy floors. This is
not a release rule I would use on the confirmation partition.

Moving the risk aggregation to recordings restores nonzero selection
in development, but these price variants add almost no qualifying
opportunities and the correction misses most that exist. Keep the
original risk checks. The next comparison should target better
predictions of realized recording-level harm on frozen paths, and
compare against a past-history-only risk proxy before adding more prices.

## Reproduction

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_recording_risk_pool.py freeze
.venv-pytorch/bin/python scripts/run_m3w_recording_risk_pool.py score
.venv-pytorch/bin/python scripts/run_m3w_recording_risk_pool.py replay
.venv-pytorch/bin/python scripts/audit_m3w_recording_risk_pool.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_recording_risk_pool.py \
    tests/test_m3w_ols_relative_gain.py tests/test_m3w_full_scene_harm_transfer.py \
    tests/test_m3w_dronecrowd*.py tests/test_m3w_recording_forecast_comparison.py
```

Native arm64 PyTorch, four threads, zero workers. No new neural fit
or CREATE job. Freeze took 26.40 seconds and scoring took 122.85 seconds
(1.93 GB maximum RSS). Local recording receipts
allow restart without repeating finished recordings. Replay uses
score summaries only. The focused suite passed 191 tests. The 30
independent confirmation sources were not accessed. Deployment,
paper, original sparse protocol, Stage5C and SMC are unchanged.

[Registration](registration.json) · [Frozen menu](freeze.json) · [Results](analysis.json) ·
[Miss attribution](attribution.json)
