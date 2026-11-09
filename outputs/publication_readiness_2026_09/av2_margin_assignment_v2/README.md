# AV2: Support Improves, but Scene Margin Training Does Not Transfer

I trained a new independent-action reference after the
[observability experiment](../av2_observability_v2/README.md)
showed that another release head could not fix the old assignment.
The forecast menu, slot count, two peak offsets and original
risk checks all stay fixed.

## Experiment

Each edit predicts five additive terms: full-grid CV gain,
OLS gain relative to the unchanged CV choice, two positive-harm
excesses, and signed easy error. A separate per-actor network
predicts the OLS fallback intercept, easy allowance and easy
membership. At inference I sum the two action values and
include the scene intercept and allowance; no interaction
term can change the values of individual edits. The scene
margin score uses a soft conjunction of five signed margins
and edited-easy support. It is not a guarantee.

I compare the frozen reference, point gains, the old-style
heuristic applied to the new costs, the scene-margin readout,
and the same margin readout trained with an additional
qualification and within-query ranking loss (the registered
primary). The two fits share initialization and supervised
component batches. This is related to
[decision-focused training](https://ojs.aaai.org/index.php/AAAI/article/view/3982)
and [predict-then-optimize](https://pubsonline.informs.org/doi/abs/10.1287/mnsc.2020.3922);
here the training term supervises qualification on enumerated
two-slot options, not the SPO loss or a differentiable solver.

The six city rotations, four-donor fit, separate calibration
city, seeds 17/29/43 and 896 held development scenarios are
unchanged. The neural candidates use the existing nested
city-excluded forecasts. The old reference forecasts replay
exactly. Calibration here tests the **reference alone** with
both motion floors and all original 2% checks, not whether
two policies can be released jointly.

## Missing-Actor Correction

An earlier vectorized capacity routine let NaN from a wholly
unlabelled actor reach the all-risk sum, even for options that
did not edit that actor. The original scalar evaluator already
handles this case: its known-error contribution is zero, and
an edited unlabelled actor still incurs its displacement debit.
The new vectorized margins reproduce those scalar checks on
every selected held two-slot forecast. The paired bound still
charges every missing trajectory/pair step as before.

Of 1,020 held two-slot views, 63 contain at least one wholly
unlabelled actor. The earlier counts reproduce when those
views are excluded from the enumeration. The corrected
hindsight reference ceiling is **382** (previously **369**);
the corrected two-path ceiling is **344** (previously **334**).
The fixed reference remains **63** eligible views and **48**
with a valid alternative; Palo Alto remains 0 fixed versus
16 possible pairs. These are seed views, not independent
scenes or deployable results.

Older AV2 enumerations and strict labels that used that helper
remain historical and are not silently overwritten. The OpenDD
headline streaming-capacity result follows a separate solver
that explicitly masks unknown rows and rechecks its witnesses;
this correction does not use or change that solver or its
source bindings.

## Held Results

| Forced policy | Known ADE | Available FDE | Conservative gain vs CV (pp) | Risk passes / 1,932 | Easy violations | Eligible / 1,020 | With a qualified alternative |
|---|---:|---:|---:|---:|---:|---:|---:|
| CV | 0.24668 | 0.57458 | 0 | No edits | No edits | Not applicable | Not applicable |
| Scene uniform | 0.24631 | 0.56954 | -1.24 | 132 | 250 | — | — |
| Frozen reference | 0.24574 | 0.56860 | -1.07 | 143 | 242 | 63 | 48 |
| Point | 0.24550 | 0.56804 | -0.99 | 103 | 147 | 34 | 30 |
| Heuristic | 0.24595 | 0.56884 | -1.19 | 165 | 283 | 76 | 53 |
| Margins | 0.24647 | 0.56941 | -1.43 | 152 | 411 | 58 | 46 |
| Supervised margins (primary) | 0.24634 | 0.56991 | -1.43 | 144 | 354 | 61 | 43 |

All matched forecasts use 2,952 edits. The primary raises
edited-easy support from 382 to 503 views, but qualification
does not improve. The city-balanced paired difference in
qualification rate versus fixed is **-0.44 pp**
[-1.79, +0.75]. The heuristic is +1.46 pp [-0.20, +2.99]
and reaches 2 eligible Palo Alto views, but the interval
includes zero and its frozen release fails both held actions.
The primary's full-grid pair gain against the fixed reference
is **-0.00438** [-0.00508, -0.00355] native units.
Intervals use 3,000 city draws after within-scenario seed
averaging. The primary's hard gain is +0.52% versus +0.56%
for fixed, and its mean scenario p95 ADE is 0.68700 versus
0.68552. OLS4, raw EqMotion and worst easy ratios are in the
[summary](summary.json); tiny easy denominators make those
worst ratios much larger than the aggregate percentages.

| Standalone reference | Passing nonempty prefixes / 1,020 | Held eligible / released |
|---|---:|---:|
| Frozen reference | 5 | No release |
| Point | 0 | No release |
| Heuristic | 8 | 0 / 2 |
| Margins | 0 | No release |
| Supervised margins | 3 | 1 / 16 |

The primary's 16 interventions all occur in Austin; the
other 15 fail original risk. Its released trajectory lower
gain is +0.00783 pp [0, +0.02349], which does not establish
transfer. I keep the fixed reference for subsequent controls
and do not train a new joint decoder on this result.
The next test should audit which predicted harm margins
are systematically optimistic across calibration cities,
then test one-sided harm estimates on the same action menu
instead of fitting another joint ranker or relaxing the
release rules.

## Reproduce

Uses the [observability and capacity artifacts](../av2_observability_v2/README.md).

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_av2_margin_assignment.py fit
.venv-pytorch/bin/python scripts/run_m3w_av2_margin_assignment.py score
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_av2*.py tests/test_m3w_interaction_frontier.py tests/test_m3w_slate_relative_gain.py
```

36 fresh fits, native arm64 PyTorch, four compute threads and
zero loader workers. The finalized v2 fit/freeze took 54.51 s
and 0.98 GB peak RSS; the held readout took 11.36 s. The v1
preflight was stopped before completing a fold to retain the
original percentage-point tolerance in the pair audit. Its
partial checkpoint is not used. Atomic checkpoints every
40 updates retain optimizer and sampler state. A complete
receipt replay took 7.22 s without new fits; 51 focused tests passed.
No CREATE job, reserved confirmation access, manuscript
edit, deployment change, Stage5C, or SMC.

[Registration](registration.json) · [Freeze](freeze.json) · [Results](summary.json)
