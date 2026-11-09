# AV2: directly ranking joint qualification does not improve release

The [count-transport audit](../av2_release_transport_v1/README.md)
showed that risk and accuracy both block the highest-ranked forecasts.
I replaced the old gain and risk regressions with a direct label:
does this fixed forecast pass the original risk checks and both
motion-baseline accuracy floors? I tested whether emphasizing the
strongest competing failure helps at a fixed intervention count.

## Matched training

Both new heads use the same causal history and projected-path features,
initialization, donor query batches and 400 updates. Queries are drawn
before options so scenes with many assignments do not dominate.
Each positive uses four rivals from other scenarios in its donor city;
two slots prefer genuinely supported failures when available. The
pointwise fit learns the joint label with class-balanced BCE. The
ranking fit adds a unit-margin softplus penalty against the highest
scored rival. These are scores, not calibrated probabilities.

For each of six city rotations and seeds 17, 29 and 43, only the four
donor cities supply training labels. The forecast, direction and action
models generating a donor's options also exclude that donor, the
calibration city and the held city. Across folds the shared pool has
4,776 donor queries and 265,296 ordered options, including 13,537
qualifying options in 1,579 positive queries. These are repeated
training counts, not independent samples.

No forecast, recipient, per-step edit magnitude or risk criterion
changes. The protocol uses eight observed and twelve predicted native
steps, two edits, CV/OLS4 accuracy floors and the original 2% checks.
Scores are frozen before applying the original calibration prefix and
scoring 340 held development scenarios (1,020 seed views).

## Same intervention count, five views per fold

| Score | Qualifies / 90 | Risk fails | Observed harm | No easy reference | Accuracy-floor fails |
|---|---:|---:|---:|---:|---:|
| Transferred cost | 31 | 45 | 35 | 10 | 48 |
| Actor support | 31 | 46 | 39 | 6 | 44 |
| Conditional harm | 28 | 45 | 24 | 21 | 46 |
| Strict pointwise | 14 | 72 | 20 | 52 | 51 |
| Strict rank | 15 | 71 | 18 | 53 | 48 |

The rank fit qualifies one more view than its matched pointwise
control (+1.11 pp, paired city 95% CI [-2.22, +4.44]) but trails
actor support by 16 views (-17.78 pp [-32.22, -4.44]). The decrease
in observed harm is not a safety gain: missing references rise from
6 to 53 and overall risk failures rise from 46 to 71.

## What changes when support is known in hindsight?

In a separate diagnostic, restrict all scores to the same 387 genuinely
supported views, then match the top-five count again. The outcome-defined
filter is not available at inference and was not used for selection.

| Score | Qualifies / 90 | Risk fails |
|---|---:|---:|
| Actor support | 33 | 42 |
| Conditional harm | 36 | 35 |
| Strict pointwise | 28 | 49 |
| Strict rank | 34 | 44 |

Within this subset, rank versus pointwise gains six qualifications
(+6.67 pp, 95% CI [0.00, +13.33]) and removes five risk failures
(-5.56 pp [-8.89, -2.22]). It still does not show an advantage over
actor support in qualification (+1.11 pp [-14.44, +15.58]) or risk.
Intervals resample the six cities 3,000 times after averaging seeds;
they are descriptive development intervals.

## Actual release

Neither new fit produces a nonempty qualifying calibration prefix
in any fold. Both use CV everywhere on held cities. This is zero
coverage, not a passed safety test. The frozen actor-support
comparator remains 8 releases, 5 qualifications and 3 risk failures;
it also is not a validated release policy. No model is promoted.

This is a separate question from jointly training a predictor and
reject option, as in [SelectiveNet](https://proceedings.mlr.press/v97/geifman19a.html):
here the forecasts and fallback are fixed, and the comparison asks
whether the selection loss retrieves joint qualification at the same
intervention count. The empirical calibration prefix is not the
finite-sample risk guarantee of [Learn then Test](https://arxiv.org/abs/2110.01052).

I would not continue tuning the same release MLP on these development
cities. The supported-only contrast isolates a limited ranking signal
against the matched control, not a gain over the existing selector.
Any further forecast-bank change should demonstrate that it increases
the number of forecasts that pass **both** accuracy and risk before
adding another learned release score. Independent external evidence
remains the paper's priority.

## Reproduce

Training and freezing took 67.84 s, 855 MB peak RSS, on native arm64
CPU with four Torch threads and no loader workers. There are 36 fits
and 14,400 updates. The 50-step atomic checkpoints preserve optimizer,
weights and RNG; running `fit` again resumes. The local heartbeat
records PID, fold and step.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_av2_release_rank.py fit
.venv-pytorch/bin/python scripts/run_m3w_av2_release_rank.py score
.venv-pytorch/bin/python scripts/audit_m3w_av2_release_rank_support.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_av2_*.py tests/test_m3w_cv_anchor_budget.py tests/test_m3w_cv_joint_feasibility.py
```

[Registration](registration.json), [freeze](freeze.json),
[calibration](calibration.json), [results](summary.json),
[supported-only audit](supported_only_audit.json).
116 focused tests passed. Scoring uses verified scalar development
receipts, not raw held futures. No new CREATE job, AV2 official
validation/test, DroneCrowd confirmation, manuscript edit, submission,
deployment, Stage5C or SMC.
