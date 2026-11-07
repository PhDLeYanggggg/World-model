# Less Disagreement With CV Does Not Yield A Better Candidate

I trained three new PCT continuations from the original 4,000-step warm
checkpoints. Each makes 300 new updates on the same 69,746 fitting rows, with
exactly the same draws and RNG states as the verified ADE and easy-only
controls. The one new term penalizes maximum deviation from the causal CV
path on wholly unobserved fitting rows. No missing position is imputed.
Existing partially observed labels and easy loss stay unchanged. No future
availability is an input at inference.

| New candidate minus easy-only continuation | Point (pp) | 95% exploratory source interval |
|---|---:|---:|
| Unknown-row radius to CV | -15.604 | [-21.847, -9.681] |
| Known-label utility | +9.851 | [-3.216, +25.036] |
| Known easy positive harm | -37.576 | [-85.773, -5.314] |
| Known hard utility | +3.462 | [-6.706, +14.053] |
| Direct paired completion-lower utility | -22.399 | [-45.471, -2.028] |

All 125,540 A rows, 14 sources, 34 recordings and three actual seeds are
retained. The radius is relative to CV, while the direct paired bound in the
last line is relative to the easy-only forecast. A trajectory can move closer
to CV yet move farther from that other learned forecast. Do not subtract two
separate CV-relative lower bounds and label the difference a paired effect.
The radius to CV falls from 43.56 to 27.95 pp and the CV-relative lower moves
from -65.48 to -40.03 pp. The new direct lower versus CV is still negative
[-73.07, -11.27] pp, and the direct lower versus ordinary ADE is also negative
(-22.26 pp [-46.29, -0.60]). The source intervals are development diagnostics.

## Original Risk And Candidate Availability

**No candidate promotion.** All 102 raw-path recording views still violate
easy completion risk; 99 violate all completion risk and 87 violate signed
easy net. The original all-conditions count is 0/102. The apparent easy
improvement in the table uses all known easy reference mass and cannot replace
the selected-reference risk requirement. The existing bank plus this path is
still label-aware feasible in only 101/102 views, with the same missing view.
No row choices from the oracle are retained for future use.

This targeted test reduces the disagreement it penalizes but supplies neither
positive conservative utility nor additional supported recordings. I do not
increase the penalty or keep tuning this PCT branch on the same exposed A
outcomes. The next decision is whether a stronger predictor plus a frozen,
independently calibrated allocator can produce a real benefit, rather than
make the prediction more similar to CV. Scientific roles for the separate
external roster remain unassigned; no new outcome is opened here.

## Reproduce The Matched Experiment

```sh
.venv-pytorch/bin/python scripts/run_m3w_missingness_aligned_forecast.py train
.venv-pytorch/bin/python scripts/run_m3w_missingness_aligned_forecast.py infer
.venv-pytorch/bin/python scripts/run_m3w_missingness_aligned_forecast.py score
.venv-pytorch/bin/python scripts/run_m3w_missingness_aligned_forecast.py replay
```

The protocol was committed at `894b07bb`. All processes exit 0: train PID
16266 (22.61 s), inference 16317 (60.30 s), score 16422 (7.03 s), replay 16478
(6.98 s). Three new models, 900 new optimizer updates, 376,620 new A forecasts.
Six matching controls and their paths are cached verified; no duplicate
control training. Checkpoints and chunk forecasts are resumable. All new
paths freeze before A labels are read; score replay is exact, not a second
full inference. Native arm64 CPU4/interop1/workers0. The original 2% rule is
unchanged. No external results, deployment, metric/seconds, Stage5C or SMC.

[Protocol](protocol.md) | [Full scalar results](analysis.json)
