# Count-Preserving Joint Forecast Exchange

**Result: no new recording passes and no measured interaction gain.**
The experiment holds the direct policy's intervention count fixed
in every query. A pair inserts one candidate for an unselected
agent and restores the motion baseline for a selected agent in
the same query. Both scores see exactly the same candidate pairs
and share fitting labels and side encoders; the joint score adds
a learned product of the side representations.

| Arm | First-query non-loss rescues | Folds with a cutoff | Paired exchanges | Total switched agents | Recording passes |
|---|---:|---:|---:|---:|---:|
| Original direct policy | not probed | not applicable | 0 | 6,952 | 2/102 |
| Additive score | 0/102 | 1/42 | 1 | 6,952 | 2/102 |
| Joint interaction score | 0/102 | 1/42 | 2 | 6,952 | 2/102 |

There are legal pairs in 84 of 102 views. The unthresholded
first-query probe applies the same one-pair budget in both
arms where a legal pair exists. Neither hits a rescue; the
zero source-bootstrap difference is not an equivalence claim.
Frozen joint versus additive paired completion-lower utility is
-0.00134 pp (95% paired source interval [-0.00367, 0.00000]).
Both leave 35 undefined easy-reference views and eight signed-easy
violations. Unchanged counts do not establish equal risk.

## Limits of the Matched Action Set

The exhaustive label-aware calculation finds seven non-loss
repairs in all 102 views, five at the first legal query. In the
35 missing-reference views, 18 have no original action to withdraw
and only 17 can form a pair. Same-query exchange could repair
**5/35**. Allowing one insertion followed by one later deletion,
while still preserving the recording count, gives **8/35** in
the union with same-query exchange. Three are later-only.
Every counted witness passes the original scoring checks; the
capacity audit exports no choices.

Earlier verified capacity on the same missing-reference views
was 22/35 for insertion alone and 33/35 for ordered insertion
with optional deletions. These policies use different action
sets: exact count matching is a useful control, but cannot be
the only learned action set for repairing missing support.
The training target is sparse even within the available set:
5,058 positive labels in 33,959,221 repeated fitting pair
occurrences (0.0149%). The negative result does not establish
whether a different learner could find the available pairs.

## Reproduction

Registered in `39a939e0` before running. With verified private
parents:

```sh
.venv-pytorch/bin/python scripts/run_m3w_atomic_swap.py capacity
.venv-pytorch/bin/python scripts/run_m3w_atomic_swap.py fit
.venv-pytorch/bin/python scripts/run_m3w_atomic_swap.py score
.venv-pytorch/bin/python scripts/run_m3w_atomic_swap.py replay
.venv-pytorch/bin/python scripts/audit_m3w_ordered_single_swap.py
```

All 42 fits completed 220 updates on native arm64 PyTorch
in about 182 seconds with four compute threads, zero loader
workers, checkpoints and heartbeats. The 102 development views
repeat 34 recordings across three frozen forecast seeds and 14
source groups. Replay reproduces the scalar score. The original
2% and reference support requirements remain unchanged.
No new independent outcomes, deployment, metric, seconds-level,
Stage5C, or SMC claim follows.
