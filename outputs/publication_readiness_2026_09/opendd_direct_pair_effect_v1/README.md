# Can I learn what two edits add beyond editing each agent alone?

The [shared-menu experiment](../opendd_slate_shared_menu_v1/README.md) showed
that the imitation-based pair score changes choices without reliably
improving forecasts. I kept the same motion forecasts, unary score and edit
budget, and trained a new pair head on the effect it is supposed to predict.

For two agents, let `E00`, `E10`, `E01` and `E11` be their relative-displacement
errors with zero, first-only, second-only and both edits. The target is
`E10 + E01 - E00 - E11`: the extra reduction from combining the edits, beyond
the sum of the two single-edit reductions. It uses steps with both future
labels, averaged and divided by a scale from the observed histories. The
head sees only the existing history and candidate-path features; the target
and its availability never enter inference. This is a forecast-error
contrast, not a claim about how changing a prediction changes real motion.

## Fixed controls

- Eight observed steps and twelve predicted steps in native OpenDD coordinates.
  Every arm searches the same analytic paths and peak slots, with 3,690 edits.
- Seven held development locations, three seeds, 21 new heads and 6,300
  optimizer updates. The frozen feature extractor and unary score exclude
  the scored location. Direct training examples are built from the donor
  prefixes and their future labels, not from a model fitted to the held site.
- A legal pair leaves at least one other agent on CV, and a training example
  needs at least four co-observed future steps. `rdb3` and `rdb7` have at most
  two agents per query and therefore no legal pair examples. The initial run
  stopped on that assumption; the corrected run excludes unsupported sites
  from pair training batches, not from policy evaluation. Of 11,590 supported
  candidate assignments, 11,221 come from `rdb1` and `rdb6`; these candidates
  can share windows and are not independent trials.
- No held-out threshold or weight selection. The added score weight is fixed
  by donor-only score scales. The original 2% risk test and missing-step debit
  remain unchanged. WEPDTOF external and DroneCrowd confirmation stay closed.

## Results

| Policy | Risk passes / 606 | Risk pass and gain over both references / 606 |
|---|---:|---:|
| Additive control | 39 | 32 |
| Imitation pair score | 41 | 34 |
| Direct pair-effect score | 41 | 34 |

The direct head changes 295 of 1,067 double-edit assignments versus additive.
All 1,556 single-edit choices are unchanged. On the separate held pair-target
probe, predicting zero beats the trained head in 14 of 15 supported
location-seed comparisons, and at all five supported locations after averaging
seeds. The other two locations have no probe target, not a zero error.

| Direct head comparison (positive is better) | Site mean | 95% location bootstrap |
|---|---:|---:|
| Observed-step gain vs. additive (pp of CV error) | +0.00541 | [+0.00090, +0.01265] |
| Conservative gain vs. additive (pp of CV error) | +0.00232 | [-0.00296, +0.01039] |
| Conservative gain vs. imitation (pp of CV error) | -0.01127 | [-0.03652, +0.00923] |
| Observed pair-relative error reduction vs. additive (native units) | +0.000031 | [-0.000050, +0.000112] |

The 606 rows are 202 recordings scored at three seeds. The intervals use
3,000 draws over six locations with matched intervention support. The
seventh location has no edits and no matched utility estimate. These are
development intervals, not independent confirmation. A small gain on observed
steps is not enough to establish an advantage under the unchanged full-grid
score or a benefit on relative geometry itself.

**Decision:** Keep the deployed fallback unchanged. Direct supervision did
not solve cross-location prediction of the added pair effect. Before spending
more compute on larger heads, test whether signed relative position and
approach velocity add information that the current magnitude-based pair
features omit, with the same menu and budget. A lack of improvement on the
held pair-target probe would argue against further tuning this head.

## Replay

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_pair_effect.py freeze
.venv-pytorch/bin/python scripts/run_m3w_pair_effect.py score
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_pair_effect.py tests/test_m3w_slate_shared_menu.py tests/test_m3w_slate_imitation.py
```

`fresh_run`: local native arm64 CPU training, inference and held development
scoring. `cached_verified`: checkpoint-based prediction replay and exact
readout replay. Four compute threads, no worker subprocesses, a PID heartbeat,
and atomic checkpoints every 50 steps. Fourteen focused tests pass, including
exact continuation after an interruption at step 50 in a synthetic test.
No manuscript, enrollment, deployment, Stage5C or SMC change.
