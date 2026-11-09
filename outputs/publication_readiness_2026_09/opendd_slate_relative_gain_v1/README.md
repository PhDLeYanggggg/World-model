# Learning the Relative-Motion Gain of a Joint Replacement

The [turn-menu experiment](../opendd_turn_intervention_v1/README.md)
showed that better assignments exist, but the joint rule rarely finds
them. I kept the menu fixed and fitted a score for the gain from
replacing the whole additive forecast. Each input still contains eight
observed positions; each output covers 12 native steps. All five policies
use the same 3,690 edit slots, displacement limits and singleton
forecasts. Only the selection in two-edit queries can change.

## What the Head Learns

Relative-motion gain is computed for every unordered pair in the
query, including pairs with one unchanged forecast. The target is the
sum of known error reductions per requested grid step, less the mean
relative-path change on missing pair steps. The latter is a
reverse-triangle-inequality debit. One head predicts the known
contribution, which reverses sign when the forecasts are exchanged.
The other predicts the missing debit, which does not change sign. Both
are bounded by the mean relative-path difference. These are predictions
of retrospective quantities, not prospective risk certificates.

I fitted the head at three seeds, leaving one of seven OpenDD locations
out at a time. Donor neural forecasts exclude the donor and outer
locations from their own fitting. Training uses OLS-nearest and random
matched-budget donor references; inference uses the frozen additive
reference and never receives future labels or their availability.
The three re-rankers all add a fixed 0.25 times their score to the
unchanged risk-aware joint utility. The analytic OLS4 proxy is a
no-training comparison. The observed-only re-ranker omits the predicted
debit at inference; it uses the same fitted head, not a second fit.

## Matched Results

There are 202 development recordings, 606 recording-seed views and 439
views with edits. Pair-gain intervals are 3,000 paired location draws,
with seed means formed inside each recording. The six estimable
locations include three with zero differences between the learned arms.
The gain columns use 10^-3 native units; ADE and available-endpoint FDE
are in native units. Strict wins require positive full-grid pair gain,
all original recording risk checks, conservative individual gains over
both CV and OLS4, and no individual loss versus the paired control.

| Policy | Risk pass | ADE / FDE | Observed gain vs additive [95% CI] | Full-grid lower gain vs additive [95% CI] | Strict wins vs additive |
|---|---:|---:|---:|---:|---:|
| Additive | 42/439 | 0.54830 / 1.25312 | reference | reference | reference |
| Risk-only joint | 46/439 | 0.54840 / 1.25340 | -0.091 [-0.238, +0.024] | -0.195 [-0.416, -0.009] | 3 |
| OLS4 proxy re-rank | 43/439 | 0.54856 / 1.25334 | +0.354 [-0.328, +1.450] | +0.196 [-0.582, +1.309] | 2 |
| Known-gain re-rank | 47/439 | 0.54825 / 1.25303 | +0.336 [+0.016, +0.907] | +0.185 [-0.199, +0.784] | 5 |
| Gain minus debit | 47/439 | 0.54826 / 1.25305 | +0.325 [+0.015, +0.905] | +0.174 [-0.223, +0.783] | 5 |

**Direct supervision helps the old joint rule.** Gain minus debit
improves its full-grid lower pair gain by **+0.321 [+0.011, +0.870]**
and its paired conservative individual gain by **+0.042 pp
[+0.002, +0.095]**. The analogous conservative pair interval against
the stronger additive control still spans zero. Five qualifying gains
over additive all come from rdb1. On 8,432 frozen option probes the
full-grid head's RMSE is 0.1013, versus 0.1079 for predicting no gain
and 0.1846 for the OLS4 proxy. These are descriptive option-level
errors, not independent samples. The analytic, known-gain and
gain-minus-debit scores change 417, 106 and 103 of the 1,067 two-edit
query-seed views relative to the old joint rule.

**The missing-debit head is not the deployment fix.** Compared
directly with known-gain re-ranking, it changes six decisions, gives
no qualifying wins, and yields -0.011 [-0.031, 0.000] conservative
pair gain in the same scaled units. Both re-rankers have identical
recording risk counts. For gain minus debit, 275 edited views fail
with observed harm alone, 61 with observed harm and an undefined
reference, and 56 with an undefined reference only. No risk check
crosses 2% solely because of the missing-label debit. Signed easy
degradation still exceeds 2% in 164 edited views.

This supports direct relative-motion supervision as a development
mechanism, not a deployable rule or an independent confirmation. The
next experiment should target actual observed harm and the missing easy
reference, not spend more capacity estimating pair-label dropout.
Matched utility and the recording-level 2% deployment check must stay
separate. The loss here regresses gains and debits; unlike
[decision-focused ranking](https://proceedings.mlr.press/v162/mandi22a.html),
it does not optimize final decision regret.

## Replay

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_opendd_slate_relative_gain.py freeze
.venv-pytorch/bin/python scripts/run_m3w_opendd_slate_relative_gain.py score
.venv-pytorch/bin/python scripts/report_m3w_opendd_slate_risk.py
```

21 new CPU fits, 400 steps each, on native arm64 with four compute
threads and no workers. Checkpoints retain weights, optimizer and sampler
state every 50 steps; the same command resumes. The local experiment
directory has a PID heartbeat and score checkpoints. The initial
smoke run failed before the first update due to a reference-axis bug;
the corrected input shape is covered by tests. No new CREATE job, no
reserved outcomes, no changed deployment, no Stage5C and no SMC.

[Registration](registration.json) | [Freeze receipt](freeze.json) |
[Readout](readout.json) | [Risk attribution and debit ablation](risk_attribution.json)
