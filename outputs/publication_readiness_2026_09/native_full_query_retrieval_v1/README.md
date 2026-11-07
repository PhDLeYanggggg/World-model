# Learning Which Recipients To Retrieve

I tested whether a past-only model could recover the extra capacity
seen when the original recipient roster was relaxed. Both rankers see
the full observed query and the same forecast bank, including the common
translation. They have the same eight-person shortlist. One uses
predicted gain for all slots; the other keeps the first half and fills
the rest with people predicted to belong to a locally feasible,
positive-gain singleton or pair.

The fit is 42 source-excluded ridge systems for the two retrieval heads,
with the same action-cost system shared by both controls in each cell.
This uses three frozen forecaster seeds, not stochastic retriever seeds.
Fitting and scoring use separate sources within exposed Moving A.
Each arm retrieves 38,688 slots, and the causal decoder proposes
9,675 switches per arm at identical query counts. The shortlists
differ in 4,774 of 4,839 seed-query cases.

| Ranker | Shortlist feasible / 102 | Shortlist positive vs. OLS4 / 102 | Switches after fixed gate | Actual recording passes / 102 |
| --- | ---: | ---: | ---: | ---: |
| Gain only | 81 | 45 | 102 | 0 |
| Gain plus local opportunity | 84 | 45 | 132 | 2 |
| Original gated policy | Not this shortlist | Not this shortlist | 90 | 3 |

The opportunity arm gains three feasible views and loses none, but the
45 positive views are identical. The gain in feasibility is +1.429 pp
with a 95% source-bootstrap interval of [0.000, +4.286]. At the same
9,675 proposed switches, the opportunity arm's direct paired
conservative ADE utility is -0.338 pp [-0.737, -0.052] versus gain-only.
The gated opportunity arm is -9.821 pp [-21.997, -0.551] versus OLS4.
It leaves easy risk undefined in 85 views. No gate threshold changes.

**Decision:** Do not promote this retriever. Moving an entire shortlist
around does not by itself improve the strong-comparator objective.
The training target checks local CV-relative gain and risk; it does not
ask which choices help a full recording beat OLS4 under a shared
budget. The next model should use recording-level strong-comparator
supervision and test it against the same gain-only control, rather than
keep reweighting local ease.

The capacity columns use exposed labels and are not attained learned
passes. The 102 views are 34 recordings across three seeds and 14
sources. Intervals use 3,000 paired source resamples after averaging
seeds. Scoring uses eight observed and twelve predicted annotation
steps in image pixels. The original 2% risk and support rules remain.
No independent data, deployment update, Stage5C or SMC.

[Protocol](protocol.md) | [Aggregate results](analysis.json) |
[Fit](fit_execution.json) | [Score](score_execution.json) |
[Replay](replay_execution.json)

```sh
.venv-pytorch/bin/python scripts/run_m3w_full_query_retrieval.py fit
.venv-pytorch/bin/python scripts/run_m3w_full_query_retrieval.py score
.venv-pytorch/bin/python scripts/run_m3w_full_query_retrieval.py replay
```

The fit freezes source-excluded decisions by seed. Score computes the
original risk and the new capacity bound. Replay reuses the bound
checkpoints and recalculates the operating scores. The private local
input, choices and checkpoints are not in Git.
