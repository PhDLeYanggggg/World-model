# Learning From Feasible Deletion Sets Does Not Recover Recording Passes

I tested supervision from verified whole-record removal sets after
the earlier moment pruner found no usable threshold. The direct
stream is unchanged. Each decision may restore the causal floor
for at most one selected action per query. The task is eight
observed and twelve forecast annotation steps in native image
coordinates on exposed Moving A development.

Each of the 42 cells regenerates direct choices on seven fitting
sources, with each producer excluding its own and all calibration/
held sources. A minimum-deletion solver witness must meet the
original risk/support requirements and nonnegative paired lower
utility; the original scorer verifies it. Across folds there are
195 verified witness occurrences and 2,684 positive deletion
occurrences, not independent records. Unsupported and infeasible
records are excluded rather than treated as negatives.

The same two-output fitted head supplies the imitation score and
predicted deletion gain. The guarded version only permits actions
with nonnegative predicted gain. Six separate donors select a
threshold under the original complete-pass, easy and utility
requirements. Unlike the earlier pruner, imitation calibrates
in 2/42 cells and the gain guard in 4/42 cells.

| Calibrated policy | Deletions | Complete passes / 102 | Signed easy >2% | Undefined easy upper | Paired lower effect [95% interval] |
|---|---:|---:|---:|---:|---:|
| Direct | 0 | 2 | 8 | 35 | Reference |
| Imitation | 75 | 2 | 8 | 35 | +0.00260 [-0.00001, +0.00780] |
| Gain guard | 75 | 1 | 7 | 36 | +0.00125 [-0.00163, +0.00552] |

Effects are percentage points of paired completion-lower utility
versus direct. We resample 14 sources 3,000 times and average three
forecast seeds within source; the 102 views repeat 34 recordings.
Neither arm meets the registered policy criterion.

## Why Donor Thresholds Do Not Transfer

Neither policy adds a held-source pass. The 75 imitation deletions
are concentrated in three views: one of the 28 failed views with a
verified non-loss repair, and two without one. The guarded version
changes nine views: four repairable failures, four without a
non-loss witness, and one existing pass. The last loses selected
easy reference and no longer passes. The lower signed-easy
violation count does not compensate for lost support.

In the fixed .10-cut diagnostic, imitation makes 2,925 deletions
and the guard 1,183. Against the previous moment ranker deleting at
identical queries and counts, paired lower effects are -0.00024 pp
[-0.01422, +0.01463] and +0.00126 pp [-0.00643, +0.00998],
respectively. Full passes are tied at two and zero within those
matched comparisons. Neither identifies a ranking benefit.

The verified teacher sets show what is possible given the whole
outcome, but the causal student largely misses the repairable
views and can lose necessary support. Adding more deletion labels
alone is not supported. The next methodological question is how to
route repairable recordings and preserve support while optimizing
the coupled decision. Decision-aware training is not new in
itself; see [Elmachtoub and Grigas](https://arxiv.org/abs/1710.08005).
This experiment imitates labels from an observed constrained
problem, rather than implementing SPO+ or claiming its guarantees.

## Reproduction

Registration `59b460b0` precedes 42 new fits. Prior forecasters
and the matched moment head are verified caches. With private
parents available, run from the repository root:

```sh
.venv-pytorch/bin/python scripts/run_m3w_pruning_teacher.py fit
.venv-pytorch/bin/python scripts/run_m3w_pruning_teacher.py score
.venv-pytorch/bin/python scripts/run_m3w_pruning_teacher.py replay
.venv-pytorch/bin/python scripts/audit_m3w_pruning_teacher_transitions.py
```

Fitting resumes from source/seed checkpoints. Score replay
reproduces the aggregate hash. The transition audit is
retrospective, not a routing feature or a change to frozen choices.
No deployment, reserve access, independent result, Stage5C or SMC.
